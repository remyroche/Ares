from __future__ import annotations

import argparse
import glob
import logging
import multiprocessing as mp
import os
import random
import traceback
from dataclasses import dataclass, replace as dc_replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numba import njit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

from extreme_price_movements.config import CFG, enable_perp_feature_keys, TEST_FEATURE_KEYS
from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    load_features_selected,
    to_panel,
)
from extreme_price_movements.periods_symbols_management import (
    EventSchema,
    SlicePlanner,
    SlicePlannerConfig,
)
from extreme_price_movements.utils import tprint

LOGGER = logging.getLogger(__name__)
_LOGGED_FAILURE_COUNTS: Dict[str, int] = {}


def _log_bounded_warning(key: str, msg: str, limit: int = 3) -> None:
    c = _LOGGED_FAILURE_COUNTS.get(key, 0)
    if c < limit:
        LOGGER.warning(msg)
    _LOGGED_FAILURE_COUNTS[key] = c + 1


def _adaptive_outer_fold_config(base_outer: Any, span_days: float) -> Any:
    span_hours = max(1.0, span_days * 24.0)
    train_h = max(12.0, span_hours * 0.50)
    valid_h = max(3.0, span_hours * 0.10)
    test_h = max(6.0, span_hours * 0.15)
    step_h = max(6.0, span_hours * 0.15)
    return base_outer.__class__(
        train_mode=base_outer.train_mode,
        train_span=pd.Timedelta(hours=train_h),
        valid_span=pd.Timedelta(hours=valid_h),
        test_span=pd.Timedelta(hours=test_h),
        step_span=pd.Timedelta(hours=step_h),
    )


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
        rank = (
            _QUOTE_PRIORITY.index(quote)
            if quote in _QUOTE_PRIORITY
            else len(_QUOTE_PRIORITY)
        )
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
def dilate_mask_by_groups_nb(
    mask: np.ndarray, group_indices: np.ndarray, duration_bars: int
) -> np.ndarray:
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


def dilate_mask_by_asset(
    mask: np.ndarray, asset_groups: Dict[int, np.ndarray], duration_bars: int
) -> np.ndarray:
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
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
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

    return (
        bars_to_peak_up,
        bars_to_peak_dn,
        speed_up,
        speed_dn,
        mono_up,
        mono_dn,
        vol_exp,
    )


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
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
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

    return (
        bars_to_peak_up,
        bars_to_peak_dn,
        speed_up,
        speed_dn,
        mono_up,
        mono_dn,
        vol_exp,
    )


def dilate_mask_by_asset_safe(
    mask: np.ndarray, asset_groups: Dict[int, np.ndarray], duration_bars: int
) -> np.ndarray:
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
def active_days_fraction_nb(
    mask: np.ndarray, day_ids: np.ndarray, n_days: int
) -> float:
    if n_days <= 0:
        return 0.0
    seen = np.zeros(n_days, dtype=np.uint8)
    n = mask.shape[0]
    for i in range(n):
        if mask[i]:
            seen[day_ids[i]] = 1
    return float(np.sum(seen)) / float(n_days)


@njit(cache=True)
def daily_event_stats_nb(
    mask: np.ndarray, day_ids: np.ndarray, n_days: int
) -> Tuple[float, float]:
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
def fold_base_rate_nb(
    mask: np.ndarray, target: np.ndarray, val_idx: np.ndarray
) -> float:
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


def active_days_fraction_safe(
    mask: np.ndarray, day_ids: np.ndarray, n_days: int
) -> float:
    if n_days <= 0:
        return 0.0
    if mask.shape[0] == 0:
        return 0.0
    active_days = np.unique(day_ids[mask])
    return float(active_days.shape[0]) / float(n_days)


def daily_event_stats_safe(
    mask: np.ndarray, day_ids: np.ndarray, n_days: int
) -> Tuple[float, float]:
    if n_days <= 0:
        return 0.0, 0.0
    counts = np.zeros(n_days, dtype=np.int32)
    if np.any(mask):
        vals, freqs = np.unique(day_ids[mask], return_counts=True)
        counts[vals.astype(np.int32)] = freqs.astype(np.int32)
    return float(np.mean(counts)), float(np.std(counts))


def fold_base_rate_safe(
    mask: np.ndarray, target: np.ndarray, val_idx: np.ndarray
) -> float:
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


def _mode_primary_target(
    mode: str, forward_returns: np.ndarray, ret_threshold: float
) -> np.ndarray:
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


def _safe_abs_ratio(numerator: float, denominator: float) -> float:
    num = _metric_or_nan(numerator)
    den = _metric_or_nan(denominator)
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
    return float(num / max(abs(den), 1e-6))


def _log_stage_snapshot(
    mode: str,
    stage: str,
    df: pd.DataFrame,
    sort_col: str,
    cols: List[str],
    top_n: int = 5,
) -> None:
    if df.empty:
        tprint(f"{stage} ({mode}): no candidates")
        return
    use_cols = [c for c in cols if c in df.columns]
    snap = df.sort_values(sort_col, ascending=False).head(top_n)[use_cols]
    tprint(f"{stage} ({mode}) top {min(top_n, len(snap))} by {sort_col}:")
    tprint(snap.to_string(index=False))


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
    mae_e = (
        float(np.nanmean(mae_arr[valid & side_mask]))
        if np.any(valid & side_mask)
        else mae_g
    )
    mae_ratio = mae_e / mae_g if mae_g > 1e-9 else 1.0

    mfe_g = float(np.nanmean(mfe_arr[valid])) if np.any(valid) else 1.0
    mfe_e = (
        float(np.nanmean(mfe_arr[valid & side_mask]))
        if np.any(valid & side_mask)
        else mfe_g
    )
    mfe_ratio = mfe_e / mfe_g if mfe_g > 1e-9 else 1.0

    return float(
        np.mean(np.clip([std_ratio, tail_ratio, mae_ratio, mfe_ratio], 0.0, 5.0))
    )


def _build_temporal_folds(
    timestamps: np.ndarray, n_samples: int, n_splits: int = 2
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Delegate fold construction to periods_symbols_management as single source of truth."""
    if n_samples < 10:
        return []
    try:
        ts = pd.to_datetime(pd.Series(timestamps), unit="s", utc=True, errors="coerce")
        if ts.isna().all():
            ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
        ts = ts.ffill().bfill()
        span_days = max(
            1.0,
            float((ts.max() - ts.min()) / pd.Timedelta(days=1))
            if ts.notna().any()
            else 1.0,
        )
        events = pd.DataFrame(
            {
                "event_id": np.arange(n_samples, dtype=np.int64),
                "symbol": np.repeat("ALL", n_samples),
                "t0": ts.to_numpy(),
                "t1": (ts + pd.Timedelta(seconds=1)).to_numpy(),
            }
        )
        cfg = SlicePlannerConfig.fast_defaults(schema=EventSchema())
        cfg = dc_replace(
            cfg,
            preset=dc_replace(
                cfg.preset,
                outer=_adaptive_outer_fold_config(cfg.preset.outer, span_days),
                inner=dc_replace(cfg.preset.inner, n_splits=max(1, int(n_splits))),
                sampling=dc_replace(
                    cfg.preset.sampling,
                    mode="full",
                    event_fraction=1.0,
                    symbol_fraction=1.0,
                ),
                symbol_policy=dc_replace(
                    cfg.preset.symbol_policy,
                    mode="all_symbols",
                    subset_fraction=1.0,
                    min_symbols_per_split=1,
                ),
            ),
            silent=True,
            min_rows_per_fold=1,
            min_symbols_per_fold=1,
        )
        bundle = SlicePlanner(cfg).build(events)
        plans = bundle["consumer_plans"]["regime_search"]
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for plan in plans:
            tr = np.asarray(plan.fit_idx, dtype=np.int32)
            va = np.asarray(plan.predict_idx, dtype=np.int32)
            if tr.size > 0 and va.size > 0:
                folds.append((tr, va))
        if folds:
            return folds
        raise ValueError(
            f"SlicePlanner failed to generate {n_splits} temporal folds from {n_samples} samples. "
            "Ensure timestamps are valid and sufficient data exists."
        )
    except Exception as e:
        _log_bounded_warning(
            "planner_fold_fallback",
            f"Planner fold delegation failed; falling back to PurgedKFold: {e}",
            limit=10,
        )
    try:
        cv = PurgedKFold(
            n_splits=n_splits, purge=43200, embargo=43200, times=timestamps
        )
        dummy = np.empty((n_samples, 1), dtype=np.float32)
        folds = list(cv.split(dummy))
        if folds:
            return [(tr.astype(np.int32), va.astype(np.int32)) for tr, va in folds]
    except Exception:
        if n_samples < 2:
            return []
        uniq_ts = np.unique(np.asarray(timestamps))
        if uniq_ts.shape[0] < 2:
            return []
        mid_ts = uniq_ts.shape[0] // 2
        train_mask = np.isin(timestamps, uniq_ts[:mid_ts])
        valid_mask = np.isin(timestamps, uniq_ts[mid_ts:])
        return [
            (
                np.flatnonzero(train_mask).astype(np.int32),
                np.flatnonzero(valid_mask).astype(np.int32),
            )
        ]
    if n_samples < 2:
        return []
    uniq_ts = np.unique(np.asarray(timestamps))
    if uniq_ts.shape[0] < 2:
        return []
    mid_ts = uniq_ts.shape[0] // 2
    train_mask = np.isin(timestamps, uniq_ts[:mid_ts])
    valid_mask = np.isin(timestamps, uniq_ts[mid_ts:])
    return [
        (
            np.flatnonzero(train_mask).astype(np.int32),
            np.flatnonzero(valid_mask).astype(np.int32),
        )
    ]


def _apply_regime_search_slice_plan(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    lookback_years: float,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]:
    """Use periods_symbols_management regime_search plans to define the optimizer sample."""
    if data.empty:
        return data, feature_dict, forward_returns
    try:
        ts = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
        span_days = max(
            1.0,
            float((ts.max() - ts.min()) / pd.Timedelta(days=1))
            if ts.notna().any()
            else 1.0,
        )
        events = pd.DataFrame(
            {
                "event_id": np.arange(data.shape[0], dtype=np.int64),
                "symbol": data["symbol"].astype(str).values,
                "t0": ts.to_numpy(),
                "t1": (ts + pd.Timedelta(seconds=1)).to_numpy(),
            }
        )
        cfg = SlicePlannerConfig.fast_defaults(schema=EventSchema())
        cfg = dc_replace(
            cfg,
            preset=dc_replace(
                cfg.preset,
                outer=_adaptive_outer_fold_config(cfg.preset.outer, span_days),
            ),
            consumer_overrides={
                **dict(cfg.consumer_overrides),
                "full_inference_lookback_years": float(lookback_years),
            },
            silent=True,
            min_rows_per_fold=1,
            min_symbols_per_fold=1,
        )
        bundle = SlicePlanner(cfg).build(events)
        consumer_plans = bundle["consumer_plans"]
        plans = consumer_plans.get("regime_search", [])
        idx_parts: List[np.ndarray] = []
        for plan in plans:
            if plan.fit_idx.size > 0:
                idx_parts.append(np.asarray(plan.fit_idx, dtype=np.int64))
            if plan.predict_idx.size > 0:
                idx_parts.append(np.asarray(plan.predict_idx, dtype=np.int64))
        if not idx_parts:
            tprint("periods/symbols regime_search slice plan produced no rows; using capped sample")
            return data, feature_dict, forward_returns
        idx = np.unique(np.concatenate(idx_parts)).astype(np.int64)
        idx.sort()
        tprint(
            "Applied periods/symbols regime_search slice plan: "
            f"rows={idx.size}/{data.shape[0]} symbols={data.iloc[idx]['symbol'].nunique()}"
        )
        data_out = data.iloc[idx].reset_index(drop=True)
        feat_out = {k: np.asarray(v)[idx] for k, v in feature_dict.items()}
        fwd_out = np.asarray(forward_returns)[idx]
        return data_out, feat_out, fwd_out
    except Exception as e:
        tprint(f"periods/symbols regime_search slice plan failed; using capped sample ({e})")
        LOGGER.warning(
            "regime_search slice-plan delegation failed; using raw sample: %s", e
        )
        return data, feature_dict, forward_returns


def _impute_and_scale_train_valid(
    X_train: np.ndarray, X_valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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
    if (
        np.unique(y[valid_mask]).shape[0] < 2
        or np.unique(preds[valid_mask]).shape[0] < 2
    ):
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
    if (
        np.unique(y[valid_mask]).shape[0] < 2
        or np.unique(preds[valid_mask]).shape[0] < 2
    ):
        return 0.5
    try:
        return float(roc_auc_score(y[valid_mask], preds[valid_mask]))
    except Exception as e:
        _log_bounded_warning("roc_auc", f"AUC scoring failed: {e}")
        return 0.5


def _lgbm_subset_auc_and_lift(
    X: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
) -> Tuple[float, float]:
    if tr_idx.shape[0] < 40 or va_idx.shape[0] < 40:
        return float("nan"), float("nan")

    y_tr = y[tr_idx]
    y_va = y[va_idx]
    if (
        np.unique(y_tr[np.isfinite(y_tr)]).shape[0] < 2
        or np.unique(y_va[np.isfinite(y_va)]).shape[0] < 2
    ):
        return float("nan"), float("nan")

    clf = LGBMClassifier(
        n_estimators=64,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )
    try:
        clf.fit(X[tr_idx], y_tr)
        preds = clf.predict_proba(X[va_idx])[:, 1].astype(np.float32)
    except Exception as e:
        _log_bounded_warning("lgbm_fit", f"LGBM fold fit failed: {e}")
        return float("nan"), float("nan")

    valid = np.isfinite(preds) & np.isfinite(y_va)
    if np.sum(valid) < 40:
        return float("nan"), float("nan")
    preds_val = preds[valid]
    y_val = y_va[valid]
    if np.unique(y_val).shape[0] < 2 or np.unique(preds_val).shape[0] < 2:
        return float("nan"), float("nan")

    try:
        auc = float(roc_auc_score(y_val, preds_val))
    except Exception as e:
        _log_bounded_warning("lgbm_auc", f"LGBM AUC scoring failed: {e}")
        auc = float("nan")

    top_q = float(np.quantile(preds_val, 0.80))
    top_mask = preds_val >= top_q
    if np.any(top_mask):
        top_rate = float(np.mean(y_val[top_mask]))
        base_rate = float(np.mean(y_val))
        lift = float(top_rate - base_rate)
    else:
        lift = float("nan")
    return auc, lift


def _lgbm_subset_cv_metrics(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    idx_subset: np.ndarray,
    n_splits: int,
    max_subset: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "auc_mean": float("nan"),
        "lift_mean": float("nan"),
        "auc_folds": np.asarray([], dtype=np.float32),
        "lift_folds": np.asarray([], dtype=np.float32),
    }
    idx = _cap_index_count(idx_subset, max_subset)
    if idx.shape[0] < 40:
        return out

    folds_local = _build_temporal_folds(
        timestamps[idx], idx.shape[0], n_splits=max(2, n_splits)
    )
    if not folds_local:
        return out

    auc_folds: List[float] = []
    lift_folds: List[float] = []
    for tr_local, va_local in folds_local:
        tr_idx = idx[tr_local]
        va_idx = idx[va_local]
        auc, lift = _lgbm_subset_auc_and_lift(X, y, tr_idx, va_idx)
        if np.isfinite(auc):
            auc_folds.append(float(auc))
        if np.isfinite(lift):
            lift_folds.append(float(lift))

    if auc_folds:
        out["auc_mean"] = float(np.mean(np.asarray(auc_folds, dtype=np.float32)))
        out["auc_folds"] = np.asarray(auc_folds, dtype=np.float32)
    if lift_folds:
        out["lift_mean"] = float(np.mean(np.asarray(lift_folds, dtype=np.float32)))
        out["lift_folds"] = np.asarray(lift_folds, dtype=np.float32)
    return out


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
    X_aug = np.concatenate([X_base, event_feature], axis=1).astype(
        np.float32, copy=False
    )
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
        if (
            np.unique(y_tr[np.isfinite(y_tr)]).shape[0] < 2
            or np.unique(y_va[np.isfinite(y_va)]).shape[0] < 2
        ):
            continue
        auc_base_fold = _classifier_oof_auc_from_folds(X_base, y_all, [(tr, va)])
        auc_aug_fold = _classifier_oof_auc_from_folds(X_aug, y_all, [(tr, va)])
        delta_fold = float(auc_aug_fold - auc_base_fold)
        fold_deltas.append(delta_fold)
        evaluated_fold_count += 1
        if delta_fold > 0:
            positive_fold_count += 1

    if evaluated_fold_count > 0:
        metrics["incremental_information_delta_auc_fold_mean"] = float(
            np.mean(np.asarray(fold_deltas, dtype=np.float32))
        )
        metrics["incremental_information_delta_auc_fold_std"] = float(
            np.std(np.asarray(fold_deltas, dtype=np.float32))
        )
        metrics["incremental_information_positive_fold_fraction"] = float(
            positive_fold_count / float(evaluated_fold_count)
        )
        metrics["incremental_information_positive_fold_count"] = float(
            positive_fold_count
        )
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

    folds_e = _build_temporal_folds(
        timestamps[idx_e], idx_e.shape[0], n_splits=n_splits
    )
    folds_ne = _build_temporal_folds(
        timestamps[idx_ne], idx_ne.shape[0], n_splits=n_splits
    )
    if not folds_e or not folds_ne:
        return np.asarray([], dtype=np.float32)

    out: List[float] = []
    for (tr_e, va_e), (tr_ne, va_ne) in zip(folds_e, folds_ne):
        auc_e = _classifier_oof_auc_from_folds(
            learn_X[idx_e], y_primary[idx_e], [(tr_e, va_e)]
        )
        auc_ne = _classifier_oof_auc_from_folds(
            learn_X[idx_ne], y_primary[idx_ne], [(tr_ne, va_ne)]
        )
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
    stability_score = (
        0.5 * max(0.0, 1.0 - std_delta / (abs(mean_delta) + 1e-9))
        + 0.5 * positive_fold_fraction
    )
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
    incr_positive = _metric_or_nan(
        row.get("incremental_information_positive_fold_fraction")
    )
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



def dispersion_to_edge(returns: np.ndarray) -> float:
    """
    DER = sigma / |mu|
    """
    mu = np.mean(returns)
    sigma = np.std(returns)

    if abs(mu) < 1e-12:
        return np.inf

    return float(sigma / abs(mu))


def fold_stability(delta_r_folds: np.ndarray) -> float:
    """
    S_r = |mean(delta_r)| / std(delta_r)
    """
    mean = np.mean(delta_r_folds)
    std = np.std(delta_r_folds)

    if std < 1e-12:
        return 0.0

    return float(abs(mean) / std)


def label_entropy(labels: np.ndarray) -> float:
    """
    Shannon entropy of discrete labels.
    """
    if len(labels) == 0:
        return 0.0
    values, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()

    return float(-(p * np.log(p + 1e-12)).sum())


def compute_net_regime_value(
    returns_E: np.ndarray,
    returns_ER: np.ndarray,
    delta_r_folds_E: np.ndarray,
    delta_r_folds_ER: np.ndarray,
    labels_E: np.ndarray,
    labels_ER: np.ndarray,
    auc_E: float,
    auc_ER: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the NetRegimeValue score.
    """
    # coverage
    coverage_ratio = len(returns_ER) / max(len(returns_E), 1)
    coverage_term = np.sqrt(coverage_ratio)

    # dispersion-to-edge
    der_E = dispersion_to_edge(returns_E)
    der_ER = dispersion_to_edge(returns_ER)
    der_ratio = der_E / der_ER if der_ER > 0 else 0
    der_ratio = float(np.clip(der_ratio, 0.5, 3.0))

    # fold stability
    sr_E = fold_stability(delta_r_folds_E)
    sr_ER = fold_stability(delta_r_folds_ER)
    sr_ratio = sr_ER / sr_E if sr_E > 0 else 1.0
    sr_ratio = float(np.clip(sr_ratio, 0.5, 3.0))

    # entropy reduction
    H_E = label_entropy(labels_E)
    H_ER = label_entropy(labels_ER)
    entropy_term = float(np.exp(np.clip(H_E - H_ER, -0.5, 0.5)))

    # AUC improvement
    auc_gain = float(np.clip(max(0.0, auc_ER - auc_E), 0.0, 0.1))
    auc_term = 1.0 + auc_gain

    score = float(coverage_term * der_ratio * sr_ratio * entropy_term * auc_term)

    diagnostics = {
        "coverage_ratio": coverage_ratio,
        "DER_E": der_E,
        "DER_ER": der_ER,
        "DER_ratio": der_ratio,
        "S_r_E": sr_E,
        "S_r_ER": sr_ER,
        "S_r_ratio": sr_ratio,
        "entropy_E": H_E,
        "entropy_ER": H_ER,
        "auc_E": auc_E,
        "auc_ER": auc_ER,
        "net_regime_value": score,
    }

    return score, diagnostics


def quick_ridge_auc(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    event_mask: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]]
) -> float:
    """
    Computes a quick out-of-sample AUC using Ridge classification logic.
    """
    if np.sum(event_mask) < 20:
        return 0.5

    y_event = labels[event_mask]
    if len(np.unique(y_event)) < 2:
        return 0.5

    # We extract unscaled data first to avoid temporal leakage
    X_event_raw = features_df[event_mask].copy().replace([np.inf, -np.inf], np.nan)
    # Mapping global indices to event indices
    global_to_local = np.full(len(event_mask), -1, dtype=np.int32)
    global_to_local[event_mask] = np.arange(np.sum(event_mask))

    oof_preds = np.full(len(y_event), np.nan, dtype=np.float32)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifier

    for tr, va in folds:
        # Get event-only indices
        tr_local = global_to_local[tr]
        tr_local = tr_local[tr_local >= 0]

        va_local = global_to_local[va]
        va_local = va_local[va_local >= 0]

        if len(tr_local) < 5 or len(va_local) < 2:
            continue

        # Impute and scale *inside* the fold using only training data
        X_tr_df = X_event_raw.iloc[tr_local].copy()
        X_va_df = X_event_raw.iloc[va_local].copy()

        tr_median = X_tr_df.median()
        X_tr_df = X_tr_df.fillna(tr_median).fillna(0.0)
        X_va_df = X_va_df.fillna(tr_median).fillna(0.0)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_df)
        X_va = scaler.transform(X_va_df)

        y_tr = y_event[tr_local]

        if len(np.unique(y_tr)) < 2:
            # Can't fit on single class
            # Just predict the mean rate of this fold
            oof_preds[va_local] = np.mean(y_tr)
            continue

        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X_tr, y_tr)

        # RidgeClassifier's decision_function returns distance to hyperplane
        preds = clf.decision_function(X_va)
        oof_preds[va_local] = preds

    valid_oof = np.isfinite(oof_preds)
    if np.sum(valid_oof) < 10 or len(np.unique(y_event[valid_oof])) < 2:
        return 0.5

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_event[valid_oof], oof_preds[valid_oof]))
    except Exception:
        return 0.5


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
    return (
        "continuation_predictability_gain"
        if _mode_is_tf(mode)
        else "reversal_predictability_gain"
    )


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
    max_event = int(
        cfg.get(
            "legacy_stage2_event_max_samples",
            cfg.get("phase2_metric_max_samples_per_class", 25_000),
        )
    )
    if max_event > 0 and idx_e.shape[0] > max_event:
        rng = np.random.RandomState(456)
        idx_e = np.sort(rng.choice(idx_e, max_event, replace=False).astype(np.int32))

    n_splits = int(cfg.get("phase2_classifier_n_splits", 3))
    y_cont = _mode_primary_target(mode, forward_returns, ret_threshold)
    y_rev = np.full(y_cont.shape[0], np.nan, dtype=np.float32)
    valid_y = np.isfinite(y_cont)
    y_rev[valid_y] = 1.0 - y_cont[valid_y]

    auc_cont_g = _classifier_oof_auc(
        learn_X[idx_g], y_cont[idx_g], timestamps[idx_g], n_splits=n_splits
    )
    auc_cont_e = _classifier_oof_auc(
        learn_X[idx_e], y_cont[idx_e], timestamps[idx_e], n_splits=n_splits
    )
    out["continuation_predictability_gain"] = float(auc_cont_e - auc_cont_g)

    auc_rev_g = _classifier_oof_auc(
        learn_X[idx_g], y_rev[idx_g], timestamps[idx_g], n_splits=n_splits
    )
    auc_rev_e = _classifier_oof_auc(
        learn_X[idx_e], y_rev[idx_e], timestamps[idx_e], n_splits=n_splits
    )
    out["reversal_predictability_gain"] = float(auc_rev_e - auc_rev_g)

    if mode in (MODE_PRICE_UP_TF, MODE_PRICE_UP_MR):
        mae_arr = shared["mae_high"]
        mfe_arr = shared["mfe_high"]
    else:
        mae_arr = shared["mae_low"]
        mfe_arr = shared["mfe_low"]

    r2_mae_g = _ridge_regression_oof_r2(
        learn_X[idx_g],
        mae_arr[idx_g],
        timestamps[idx_g],
        clip_q=0.98,
        n_splits=n_splits,
    )
    r2_mae_e = _ridge_regression_oof_r2(
        learn_X[idx_e],
        mae_arr[idx_e],
        timestamps[idx_e],
        clip_q=0.98,
        n_splits=n_splits,
    )
    out["MAE_predictability_gain"] = float(r2_mae_e - r2_mae_g)

    r2_mfe_g = _ridge_regression_oof_r2(
        learn_X[idx_g],
        mfe_arr[idx_g],
        timestamps[idx_g],
        clip_q=0.98,
        n_splits=n_splits,
    )
    r2_mfe_e = _ridge_regression_oof_r2(
        learn_X[idx_e],
        mfe_arr[idx_e],
        timestamps[idx_e],
        clip_q=0.98,
        n_splits=n_splits,
    )
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
        lo = (
            np.quantile(y_tr, 1.0 - clip_q).astype(np.float32)
            if np.any(y_tr < 0)
            else 0.0
        )
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
        lo = (
            np.quantile(y_tr, 1.0 - clip_q).astype(np.float32)
            if np.any(y_tr < 0)
            else 0.0
        )
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


def _single_feature_fold_r2(
    x: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    clip_q: float = 0.98,
) -> float:
    if tr_idx.shape[0] < 10 or va_idx.shape[0] < 10:
        return float("nan")

    y_tr = y[tr_idx].astype(np.float32, copy=True)
    y_va = y[va_idx].astype(np.float32, copy=False)
    hi = np.quantile(y_tr, clip_q).astype(np.float32)
    lo = (
        np.quantile(y_tr, 1.0 - clip_q).astype(np.float32)
        if np.any(y_tr < 0)
        else 0.0
    )
    y_tr_clip = np.clip(y_tr, lo, hi).astype(np.float32)

    X_tr, X_va = _impute_and_scale_train_valid(
        x[tr_idx].reshape(-1, 1), x[va_idx].reshape(-1, 1)
    )
    reg = Ridge(alpha=1.0)
    try:
        reg.fit(X_tr, y_tr_clip)
        preds = reg.predict(X_va).astype(np.float32)
    except Exception as e:
        _log_bounded_warning(
            "single_feature_ridge_fit", f"Single-feature ridge fit failed: {e}"
        )
        return float("nan")

    valid = np.isfinite(preds) & np.isfinite(y_va)
    if np.sum(valid) < 10:
        return float("nan")
    ssr = float(np.sum((y_va[valid] - preds[valid]) ** 2))
    sst = float(np.sum((y_va[valid] - np.mean(y_va[valid])) ** 2))
    if sst < 1e-9:
        return float("nan")
    return float(1.0 - ssr / sst)


def _cap_index_count(idx: np.ndarray, max_count: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int32)
    if idx.shape[0] <= max_count:
        return idx
    pos = np.linspace(0, idx.shape[0] - 1, num=max_count, dtype=np.int32)
    return idx[pos]


def _ridge_subset_fold_metrics(
    X: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    clip_q: float = 0.98,
) -> Tuple[float, float]:
    if tr_idx.shape[0] < 20 or va_idx.shape[0] < 20:
        return float("nan"), float("nan")

    y_tr = y[tr_idx].astype(np.float32, copy=True)
    y_va = y[va_idx].astype(np.float32, copy=False)
    hi = np.quantile(y_tr, clip_q).astype(np.float32)
    lo = (
        np.quantile(y_tr, 1.0 - clip_q).astype(np.float32)
        if np.any(y_tr < 0)
        else 0.0
    )
    y_tr_clip = np.clip(y_tr, lo, hi).astype(np.float32)

    X_tr, X_va = _impute_and_scale_train_valid(X[tr_idx], X[va_idx])
    reg = Ridge(alpha=1.0)
    try:
        reg.fit(X_tr, y_tr_clip)
        preds = reg.predict(X_va).astype(np.float32)
    except Exception as e:
        _log_bounded_warning("subset_ridge_fit", f"Subset ridge fit failed: {e}")
        return float("nan"), float("nan")

    valid = np.isfinite(preds) & np.isfinite(y_va)
    if np.sum(valid) < 20:
        return float("nan"), float("nan")
    y_val = y_va[valid]
    preds_val = preds[valid]
    sst = float(np.sum((y_val - np.mean(y_val)) ** 2))
    if sst < 1e-9:
        r2 = float("nan")
    else:
        ssr = float(np.sum((y_val - preds_val) ** 2))
        r2 = float(1.0 - ssr / sst)

    top_q = float(np.quantile(preds_val, 0.80))
    bot_q = float(np.quantile(preds_val, 0.20))
    top_mask = preds_val >= top_q
    bot_mask = preds_val <= bot_q
    if np.any(top_mask) and np.any(bot_mask):
        spread = float(np.mean(y_val[top_mask]) - np.mean(y_val[bot_mask]))
    else:
        spread = float("nan")
    return r2, spread


def _extract_learnability_features(
    feature_dict: Dict[str, np.ndarray], n_samples: int
) -> np.ndarray:
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


def _flatten_wide_frame(
    df: pd.DataFrame, index: pd.Index, columns: pd.Index
) -> np.ndarray:
    return (
        df.reindex(index=index, columns=columns)
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(-1)
    )


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


def _sample_history_fraction_mask(
    day_ids: np.ndarray, frac: float, seed: int = 42
) -> np.ndarray:
    uniq_days = np.unique(day_ids)
    selected_days = list(
        set(_rng_sample_fraction(uniq_days.tolist(), frac=frac, seed=seed))
    )
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

    uniq_ts, first_idx, counts = np.unique(
        timestamps, return_index=True, return_counts=True
    )
    if require_rectangular and np.unique(counts).shape[0] > 1:
        raise ValueError(
            "Rectangular panel required, but per-timestamp row counts differ."
        )

    ref_order = None
    for pos, st in enumerate(first_idx):
        c = counts[pos]
        curr = symbols[st : st + c]
        if ref_order is None:
            ref_order = curr
        elif require_rectangular and (
            c != ref_order.shape[0] or np.any(curr != ref_order)
        ):
            raise ValueError(
                "Rectangular panel required, but symbol ordering differs by timestamp."
            )


def _safe_param_to_string(param: Any) -> str:
    if isinstance(param, tuple):
        return str(tuple(float(x) for x in param))
    return (
        str(float(param))
        if isinstance(param, (int, float, np.integer, np.floating))
        else str(param)
    )


def _build_candidate_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Builds the candidate grid using robust-z feature thresholds.
    Replaces old lookback/abs/duration grids.
    """
    lookbacks_h = [4, 8, 12]
    thresholds = [1.4, 1.6, 1.8, 2.0]

    # Define features and their families + directions
    feature_specs = [
        # Volatility Expansion (positive magnitude)
        ("hl_range", "volatility_expansion", "magnitude_positive_only"),
        ("intrabar_range_atr", "volatility_expansion", "magnitude_positive_only"),

        # Compression Transition
        ("compression_expansion_transition", "compression_transition", "magnitude_positive_only"),

        # Volume (positive magnitude)
        ("volume_robust_z", "volume", "magnitude_positive_only"),

        # Breakout / Structure
        ("breakout_distance_up_atr", "structure", "magnitude_positive_only"),
        ("breakout_distance_down_atr", "structure", "magnitude_positive_only"),

        # Stretch Location
        ("distance_from_ema_atr", "stretch_location", "directional_two_sided"),
        ("distance_from_vwap_atr", "stretch_location", "directional_two_sided"),

        # Momentum
        ("atr_normalized_trailing_return", "momentum", "directional_two_sided"),
        ("short_minus_long_momentum", "momentum", "directional_two_sided"),
        ("slope_change", "momentum", "directional_two_sided"),

        # Path Structure
        ("path_efficiency_ratio", "path_structure", "directional_two_sided"),
    ]

    candidates = []
    for f_base, family, d_type in feature_specs:
        for lb in lookbacks_h:
            fname = f"{f_base}_{lb}h"

            for th in thresholds:
                if d_type in ("magnitude_positive_only", "directional_two_sided"):
                    candidates.append({
                        "feature_base": f_base,
                        "lookback_h": lb,
                        "feature_name": fname,
                        "family": family,
                        "direction": "gt",
                        "threshold": th
                    })

                if d_type == "directional_two_sided":
                    candidates.append({
                        "feature_base": f_base,
                        "lookback_h": lb,
                        "feature_name": fname,
                        "family": family,
                        "direction": "lt",
                        "threshold": -th
                    })

    return candidates

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



@njit
def _rolling_robust_z_1d(x: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full_like(x, np.nan)
    for i in range(window - 1, n):
        w = x[i - window + 1: i + 1]
        valid = w[np.isfinite(w)]
        if len(valid) > 0:
            med = np.median(valid)
            mad = np.median(np.abs(valid - med))

            if mad < 1e-12:
                # Fallback to standard deviation if MAD is extremely small (constant area)
                if len(valid) > 1:
                    std = np.std(valid)
                    denom = std if std > 1e-12 else 1e-6
                else:
                    denom = 1e-6
            else:
                denom = 1.4826 * mad + 1e-6

            z = (x[i] - med) / denom
            # Clamp to prevent explosion
            out[i] = max(min(z, 10.0), -10.0)
    return out

def compute_robust_z_for_groups(x: np.ndarray, asset_groups: Dict[int, np.ndarray], window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for _, idxs in asset_groups.items():
        out[idxs] = _rolling_robust_z_1d(x[idxs], window)
    return out

def _compute_z_cache(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ret_1: np.ndarray,
    vol_g: np.ndarray,
    asset_groups: Dict[int, np.ndarray],
    z: int,
    bph: int,
    volume: Optional[np.ndarray] = None,
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

        # New features
        "hl_range": np.zeros(n, dtype=np.float32),
        "intrabar_range_atr": np.zeros(n, dtype=np.float32),
        "compression_expansion_transition": np.zeros(n, dtype=np.float32),
        "volume_robust_z": np.zeros(n, dtype=np.float32),
        "breakout_distance_up_atr": np.zeros(n, dtype=np.float32),
        "breakout_distance_down_atr": np.zeros(n, dtype=np.float32),
        "distance_from_ema_atr": np.zeros(n, dtype=np.float32),
        "distance_from_vwap_atr": np.zeros(n, dtype=np.float32),
        "atr_normalized_trailing_return": np.zeros(n, dtype=np.float32),
        "short_minus_long_momentum": np.zeros(n, dtype=np.float32),
        "slope_change": np.zeros(n, dtype=np.float32),
        "path_efficiency_ratio": np.zeros(n, dtype=np.float32),
    }

    for _, idxs in asset_groups.items():
        ast_high = high[idxs]
        ast_low = low[idxs]
        ast_close = close[idxs]
        ast_ret = ret_1[idxs]
        ast_vol = vol_g[idxs]

        hv, hi = rolling_max_index_nb(ast_high, z)
        lv, li = rolling_min_index_nb(ast_low, z)
        st_idx = np.maximum(0, np.arange(ast_close.shape[0], dtype=np.int32) - z + 1)
        st_px = ast_close[st_idx]

        um = np.where(st_px > 1e-9, (hv - st_px) / st_px, 0.0).astype(np.float32)
        dm = np.where(st_px > 1e-9, (st_px - lv) / st_px, 0.0).astype(np.float32)
        rm = np.where(st_px > 1e-9, (hv - lv) / st_px, 0.0).astype(np.float32)

        b_u, b_d, s_u, s_d, m_u, m_d, v_e = compute_impulse_coherence_nb(
            ast_ret, ast_vol, hv, lv, st_px, hi, li, st_idx, z
        )

        cache["up"][idxs] = um
        cache["dn"][idxs] = dm
        cache["rng"][idxs] = rm
        cache["std_up"][idxs] = rolling_std_nb(um, 30 * 24 * bph).astype(np.float32)
        cache["std_dn"][idxs] = rolling_std_nb(dm, 30 * 24 * bph).astype(np.float32)
        cache["b_up"][idxs] = b_u
        cache["b_dn"][idxs] = b_d
        cache["s_up"][idxs] = s_u
        cache["s_dn"][idxs] = s_d
        cache["m_up"][idxs] = m_u
        cache["m_dn"][idxs] = m_d
        cache["v_exp"][idxs] = v_e

        # --- New Features ---
        window_14d = 14 * 24 * bph

        # 1. Volatility / Range
        ast_hl_range = ast_high - ast_low
        cache["hl_range"][idxs] = _rolling_robust_z_1d(ast_hl_range, window_14d)

        # Use ast_vol (which is approx ATR percent) or calculate explicitly if needed.
        # Vol_g is the 14-day ATR pct. So intrabar_range_atr can be approximated.
        ast_intrabar_range_atr = np.where(ast_vol > 1e-6, (ast_high - ast_low) / (ast_close * ast_vol), 0.0)
        cache["intrabar_range_atr"][idxs] = _rolling_robust_z_1d(ast_intrabar_range_atr, window_14d)

        ast_bollinger_width = rolling_std_nb(ast_close, 20) / np.maximum(ast_close, 1e-6)
        ast_range_spike = ast_intrabar_range_atr
        # For simplicity, compression_expansion_transition is just range_spike / (bollinger_width + eps)
        ast_comp_exp = ast_range_spike / np.maximum(ast_bollinger_width, 1e-6)
        cache["compression_expansion_transition"][idxs] = _rolling_robust_z_1d(ast_comp_exp, window_14d)

        # 2. Volume
        if volume is not None:
            ast_vol_raw = volume[idxs]
        else:
            ast_vol_raw = np.ones_like(ast_close)
        cache["volume_robust_z"][idxs] = _rolling_robust_z_1d(ast_vol_raw, window_14d)

        # 3. Breakout / Structure
        # distance from trailing max high
        ast_trailing_high, _ = rolling_max_index_nb(ast_high, z)
        ast_breakout_up = (ast_close - np.roll(ast_trailing_high, 1)) / np.maximum(ast_close * ast_vol, 1e-6)
        cache["breakout_distance_up_atr"][idxs] = _rolling_robust_z_1d(ast_breakout_up, window_14d)

        ast_trailing_low, _ = rolling_min_index_nb(ast_low, z)
        ast_breakout_dn = (np.roll(ast_trailing_low, 1) - ast_close) / np.maximum(ast_close * ast_vol, 1e-6)
        cache["breakout_distance_down_atr"][idxs] = _rolling_robust_z_1d(ast_breakout_dn, window_14d)

        # 3.5 Stretch Location
        # distance_from_ema_atr
        # EMA over z bars. Simple SMA proxy if EMA is too slow inside numba, or we can use convolve.
        # Let's use SMA as a robust proxy for EMA over 'z' window to keep it vectorized here.
        sma_z = np.convolve(ast_close, np.ones(z)/z, mode='valid')
        sma_z = np.concatenate([np.full(z-1, np.nan), sma_z])
        ast_dist_ema = (ast_close - sma_z) / np.maximum(ast_close * ast_vol, 1e-6)
        cache["distance_from_ema_atr"][idxs] = _rolling_robust_z_1d(ast_dist_ema, window_14d)

        # distance_from_vwap_atr
        # VWAP over z bars = sum(close * volume) / sum(volume)
        if volume is not None:
            vol_w = volume[idxs]
        else:
            vol_w = np.ones_like(ast_close)

        sum_vol_z = np.convolve(vol_w, np.ones(z), mode='valid')
        sum_vol_z = np.concatenate([np.full(z-1, np.nan), sum_vol_z])

        sum_pv_z = np.convolve(ast_close * vol_w, np.ones(z), mode='valid')
        sum_pv_z = np.concatenate([np.full(z-1, np.nan), sum_pv_z])

        vwap_z = sum_pv_z / np.maximum(sum_vol_z, 1e-6)
        ast_dist_vwap = (ast_close - vwap_z) / np.maximum(ast_close * ast_vol, 1e-6)
        cache["distance_from_vwap_atr"][idxs] = _rolling_robust_z_1d(ast_dist_vwap, window_14d)

        # 4. Momentum
        ast_trailing_ret = (ast_close - np.roll(ast_close, z)) / np.maximum(np.roll(ast_close, z), 1e-6)
        ast_atr_norm_ret = ast_trailing_ret / np.maximum(ast_vol, 1e-6)
        cache["atr_normalized_trailing_return"][idxs] = _rolling_robust_z_1d(ast_atr_norm_ret, window_14d)

        # Short minus long
        short_ret = (ast_close - np.roll(ast_close, max(1, z//3))) / np.maximum(np.roll(ast_close, max(1, z//3)), 1e-6)
        cache["short_minus_long_momentum"][idxs] = _rolling_robust_z_1d(short_ret - ast_trailing_ret, window_14d)

        # Slope change (diff of rolling return)
        ast_slope_change = ast_trailing_ret - np.roll(ast_trailing_ret, 1)
        cache["slope_change"][idxs] = _rolling_robust_z_1d(ast_slope_change, window_14d)

        # 5. Path Structure
        # path efficiency = net move / sum of abs moves
        ast_abs_moves = np.abs(ast_close - np.roll(ast_close, 1))
        # We need a rolling sum for path efficiency. Safe rolling sum:
        rolling_abs_moves = np.convolve(ast_abs_moves, np.ones(z, dtype=int), 'valid')
        rolling_abs_moves = np.concatenate([np.full(z-1, np.nan), rolling_abs_moves])

        ast_path_eff = np.where(rolling_abs_moves > 1e-6, (ast_close - np.roll(ast_close, z)) / rolling_abs_moves, 0.0)
        cache["path_efficiency_ratio"][idxs] = _rolling_robust_z_1d(ast_path_eff, window_14d)


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

    tprint(f"Capping at {max_rows} rows (contiguous tail slice)...")
    start_idx = max(0, data.shape[0] - max_rows)
    indices = np.arange(start_idx, data.shape[0])

    data_capped = data.iloc[indices].reset_index(drop=True)
    forward_capped = forward_returns[indices]
    feature_dict_capped = {k: v[indices] for k, v in feature_dict.items()}
    return data_capped, feature_dict_capped, forward_capped


def _materialize_layer_runtime_cfg(
    cfg: Dict[str, Any], layer_name: str
) -> Dict[str, Any]:
    runtime_cfg = dict(cfg)
    if layer_name == "layer1":
        runtime_cfg["mask_opt_max_rows"] = int(
            cfg.get("layer1_mask_opt_max_rows", cfg.get("mask_opt_max_rows", 10_000))
        )
        runtime_cfg["phase1_classifier_max_samples_per_class"] = int(
            cfg.get(
                "layer1_phase1_classifier_max_samples_per_class",
                cfg.get("phase1_classifier_max_samples_per_class", 15_000),
            )
        )
        runtime_cfg["phase2_metric_max_samples_per_class"] = int(
            cfg.get(
                "layer1_phase2_metric_max_samples_per_class",
                cfg.get("phase2_metric_max_samples_per_class", 25_000),
            )
        )
        runtime_cfg["phase1_classifier_n_splits"] = int(
            cfg.get("layer1_phase1_classifier_n_splits", 2)
        )
        runtime_cfg["phase2_classifier_n_splits"] = int(
            cfg.get("layer1_phase2_classifier_n_splits", 2)
        )
        runtime_cfg["phase2_metric_fold_splits"] = int(
            cfg.get("layer1_phase2_metric_fold_splits", 2)
        )
        runtime_cfg["incremental_information_n_splits"] = int(
            cfg.get("layer1_incremental_information_n_splits", 2)
        )
    else:
        runtime_cfg["phase1_classifier_n_splits"] = int(
            cfg.get("phase1_classifier_n_splits", 2)
        )
        runtime_cfg["phase2_classifier_n_splits"] = int(
            cfg.get("phase2_classifier_n_splits", 2)
        )
        runtime_cfg["phase2_metric_fold_splits"] = int(
            cfg.get("phase2_metric_fold_splits", 3)
        )
        runtime_cfg["incremental_information_n_splits"] = int(
            cfg.get("incremental_information_n_splits", 3)
        )
    runtime_cfg["regime_score_layer"] = layer_name
    return runtime_cfg


def _rescale_mode_gates_for_sample_size(
    cfg: Dict[str, Any], n_rows: int
) -> Dict[str, Any]:
    runtime_cfg = dict(cfg)
    if n_rows <= 0:
        return runtime_cfg

    # Per-bucket masks are sparse on capped samples; fixed 5k-event gates are too high.
    target_event_density = float(
        runtime_cfg.get("mask_opt_target_event_density", 0.012)
    )
    min_events_floor = int(runtime_cfg.get("mask_opt_min_events_floor", 150))
    scaled_min_events = max(min_events_floor, int(round(n_rows * target_event_density)))

    base_active = float(runtime_cfg.get("phase2_min_active_days_fraction", 0.80))
    active_days_floor = float(runtime_cfg.get("mask_opt_min_active_days_floor", 0.25))
    scaled_active_days = min(
        base_active, max(active_days_floor, base_active * np.sqrt(n_rows / 300_000.0))
    )

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
    candidate: Dict[str, Any],
    zc: Dict[str, np.ndarray],
    asset_groups: Optional[Dict[int, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:

    f_base = candidate["feature_base"]
    direction = candidate["direction"]
    threshold = candidate["threshold"]

    if f_base not in zc:
        raise ValueError(f"Feature {f_base} not found in zc cache!")

    feature_vals = zc[f_base]

    mask_h = np.zeros(feature_vals.shape[0], dtype=bool)
    mask_l = np.zeros(feature_vals.shape[0], dtype=bool)

    # We evaluate directions depending on the prompt logic.
    # Note: "directional_two_sided" tests both positive and negative bounds independently in separate candidates!
    # But mask_h and mask_l usually denote "Price Up" and "Price Down" events.
    # For a feature like `atr_normalized_trailing_return` > 1.6:
    # Does this mean price went UP? Yes. So it goes to mask_h.
    # For `atr_normalized_trailing_return` < -1.6:
    # It means price went DOWN. It goes to mask_l.

    valid_mask = np.isfinite(feature_vals)

    if candidate["family"] in ("volatility_expansion", "compression_transition", "volume"):
        # These are magnitude / expansion indicators. They don't have inherent up/down logic themselves.
        # We route them to mask_h or mask_l based on the concurrent price action direction.
        if direction == "gt":
            trigger = valid_mask & (feature_vals >= threshold)
            up_move = zc.get("up", np.zeros_like(feature_vals))
            dn_move = zc.get("dn", np.zeros_like(feature_vals))
            mask_h = trigger & (up_move >= dn_move)
            mask_l = trigger & (dn_move >= up_move)

    elif candidate["family"] == "structure":
        # breakout_distance_up_atr > 1.4 -> price went UP -> mask_h
        # breakout_distance_down_atr > 1.4 (since it's computed as roll(low) - close) -> price went DOWN -> mask_l
        if f_base == "breakout_distance_up_atr" and direction == "gt":
            mask_h = valid_mask & (feature_vals >= threshold)
        elif f_base == "breakout_distance_down_atr" and direction == "gt":
            mask_l = valid_mask & (feature_vals >= threshold)

    elif candidate["family"] in ("momentum", "stretch_location", "path_structure"):
        # Directional two-sided
        if direction == "gt":
            mask_h = valid_mask & (feature_vals >= threshold)
        elif direction == "lt":
            mask_l = valid_mask & (feature_vals <= threshold)

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
        return np.nan_to_num(
            np.asarray(feature_dict[name], dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

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
        score = (
            0.35 * (-impulse) + 0.20 * vol + 0.20 * rng - 0.15 * rev - 0.10 * entropy
        )
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
    atr = np.asarray(
        feature_dict.get("atr", np.ones_like(close, dtype=np.float32)), dtype=np.float32
    )

    # ids
    symbol_uniques, symbol_codes = np.unique(symbols, return_inverse=True)
    symbol_codes = symbol_codes.astype(np.int32)
    day_ids, n_days = _build_day_ids(timestamps)
    timestamp_ids, n_timestamps = _build_timestamp_ids(timestamps)
    regime_source = np.asarray(
        feature_dict.get("vol_regime_z", np.zeros_like(close, dtype=np.float32)),
        dtype=np.float32,
    )
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
        vol_g[idxs] = rolling_std_nb(r, 30 * 24 * bph).astype(np.float32)

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
    # NaN => no full forward horizon available for this row.
    mfe_atr = np.full(n, np.nan, dtype=np.float32)
    mae_atr = np.full(n, np.nan, dtype=np.float32)

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
            mfe_atr[idxs[i]] = (np.max(h_sl) - c_i) / atr_i
            mae_atr[idxs[i]] = (c_i - np.min(l_sl)) / atr_i
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
        "mfe_atr": mfe_atr,
        "mae_atr": mae_atr,
        "learn_X": learn_X,
        "day_ids": day_ids,
        "n_days": n_days,
        "timestamp_ids": timestamp_ids,
        "n_timestamps": n_timestamps,
        "regime_ids": regime_ids,
        "folds": folds,
        "tbm_geometry_cache": {},
        "z_grid": sorted(
            set(int(z * bph) for z in cfg.get("z_hours_grid", [6, 10, 16]))
        ),
        "candidate_grid": _build_candidate_grid(cfg),
        "volume": data["volume"].values if "volume" in data.columns else None,
    }


# =============================================================================
# PHASE 1 + PHASE 2
# =============================================================================


def _phase1_subsample_indices(
    shared: Dict[str, Any], cfg: Dict[str, Any], seed: int = 42
) -> np.ndarray:
    symbol_codes = shared["symbol_codes"]
    n_total = symbol_codes.shape[0]

    max_phase1_rows = int(cfg.get("phase1_max_subsample_rows", 20_000))

    if n_total <= max_phase1_rows:
        return np.ones(n_total, dtype=bool)

    rng = np.random.RandomState(seed)
    indices = rng.choice(n_total, size=max_phase1_rows, replace=False)
    result = np.zeros(n_total, dtype=bool)
    result[indices] = True
    return result


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
        "asset_groups": _build_asset_groups_from_codes(
            symbol_codes_local, shared["symbol_uniques"].shape[0]
        ),
    }
    phase_local["n_days"] = (
        int(np.max(phase_local["day_ids"]) + 1)
        if phase_local["day_ids"].shape[0] > 0
        else 0
    )
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

    idx_e, idx_ne = _balanced_sample_indices(
        idx_e, idx_ne, max_samples_per_class, seed=42
    )
    auc_ne = _classifier_oof_auc(
        learn_X[idx_ne], y_global[idx_ne], timestamps[idx_ne], n_splits=n_splits
    )
    auc_e = _classifier_oof_auc(
        learn_X[idx_e], y_global[idx_e], timestamps[idx_e], n_splits=n_splits
    )
    return float(auc_e - auc_ne)


def _compute_phase3_feature_learnability(
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    side_mask: np.ndarray,
    mode: str,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    min_pos_frac = float(cfg.get("min_feature_positive_fold_fraction", 0.60))
    top_k = int(cfg.get("feature_learnability_top_k", 10))
    y = _signed_mode_return(mode, shared["forward_returns"]).astype(np.float32)
    n = y.shape[0]
    regime = side_mask.astype(bool)

    surviving_lifts: List[float] = []
    surviving_pos_frac: List[float] = []
    per_feature_top: List[Tuple[str, float]] = []

    for fname, arr in feature_dict.items():
        x = np.asarray(arr, dtype=np.float32)
        if x.shape[0] != n:
            continue
        fold_lifts: List[float] = []

        for tr, va in folds:
            valid = np.isfinite(x) & np.isfinite(y)
            tr_reg = tr[regime[tr] & valid[tr]]
            va_reg = va[regime[va] & valid[va]]
            tr_non = tr[(~regime[tr]) & valid[tr]]
            va_non = va[(~regime[va]) & valid[va]]
            tr_full = tr[valid[tr]]
            va_full = va[valid[va]]

            if (
                tr_reg.shape[0] < 10
                or va_reg.shape[0] < 10
                or tr_non.shape[0] < 10
                or va_non.shape[0] < 10
            ):
                continue

            reg_r2 = _single_feature_fold_r2(x, y, tr_reg, va_reg)
            non_r2 = _single_feature_fold_r2(x, y, tr_non, va_non)
            full_r2 = _single_feature_fold_r2(x, y, tr_full, va_full)
            baseline_r2 = max(non_r2, full_r2)
            if not np.isfinite(reg_r2) or not np.isfinite(baseline_r2):
                continue
            fold_lifts.append(float(reg_r2 - baseline_r2))
        if not fold_lifts:
            continue
        folds_arr = np.asarray(fold_lifts, dtype=np.float32)
        mean_lift = float(np.mean(folds_arr))
        pos_frac = float(np.mean(folds_arr > 0.0))
        if mean_lift > 0.0 and pos_frac >= min_pos_frac:
            surviving_lifts.append(mean_lift)
            surviving_pos_frac.append(pos_frac)
            per_feature_top.append((fname, mean_lift))

    if surviving_lifts:
        top_vals = np.sort(np.asarray(surviving_lifts, dtype=np.float32))[-top_k:]
        gain = float(np.mean(top_vals))
        top_pairs = sorted(per_feature_top, key=lambda t: t[1], reverse=True)[:top_k]
    else:
        gain = 0.0
        top_pairs = []
    return {
        "feature_learnability_gain": np.float32(gain),
        "top_feature_lifts": ";".join([f"{k}:{v:.6f}" for k, v in top_pairs]),
        "feature_positive_fold_fraction": np.float32(
            float(np.mean(surviving_pos_frac)) if surviving_pos_frac else 0.0
        ),
    }


def _compute_conditional_predictability_metrics(
    shared: Dict[str, Any],
    side_mask: np.ndarray,
    mode: str,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
) -> Dict[str, np.float32]:
    y = _signed_mode_return(mode, shared["forward_returns"]).astype(np.float32)
    X = np.asarray(shared["learn_X"], dtype=np.float32)
    valid = np.isfinite(y)
    regime = side_mask.astype(bool) & valid
    nonregime = (~side_mask.astype(bool)) & valid
    max_subset = int(cfg.get("phase2_metric_max_samples_per_class", 25_000))

    gain_folds: List[float] = []
    spread_folds: List[float] = []
    regime_r2_vals: List[float] = []
    baseline_r2_vals: List[float] = []

    for tr, va in folds:
        tr_reg = _cap_index_count(tr[regime[tr]], max_subset)
        va_reg = _cap_index_count(va[regime[va]], max_subset)
        tr_non = _cap_index_count(tr[nonregime[tr]], max_subset)
        va_non = _cap_index_count(va[nonregime[va]], max_subset)
        tr_full = _cap_index_count(tr[valid[tr]], max_subset)
        va_full = _cap_index_count(va[valid[va]], max_subset)

        reg_r2, reg_spread = _ridge_subset_fold_metrics(X, y, tr_reg, va_reg)
        non_r2, _ = _ridge_subset_fold_metrics(X, y, tr_non, va_non)
        full_r2, _ = _ridge_subset_fold_metrics(X, y, tr_full, va_full)
        baseline_r2 = max(
            _metric_or_nan(non_r2),
            _metric_or_nan(full_r2),
        )
        if not np.isfinite(reg_r2) or not np.isfinite(baseline_r2):
            continue

        gain_folds.append(float(reg_r2 - baseline_r2))
        regime_r2_vals.append(float(reg_r2))
        baseline_r2_vals.append(float(baseline_r2))
        if np.isfinite(reg_spread):
            spread_folds.append(float(reg_spread))

    gain_arr = np.asarray(gain_folds, dtype=np.float32)
    spread_arr = np.asarray(spread_folds, dtype=np.float32)
    regime_arr = np.asarray(regime_r2_vals, dtype=np.float32)
    baseline_arr = np.asarray(baseline_r2_vals, dtype=np.float32)

    return {
        "conditional_predictability_gain": np.float32(
            float(np.mean(gain_arr)) if gain_arr.size > 0 else 0.0
        ),
        "conditional_predictability_positive_fold_fraction": np.float32(
            float(np.mean(gain_arr > 0.0)) if gain_arr.size > 0 else 0.0
        ),
        "conditional_predictability_regime_r2": np.float32(
            float(np.mean(regime_arr)) if regime_arr.size > 0 else 0.0
        ),
        "conditional_predictability_baseline_r2": np.float32(
            float(np.mean(baseline_arr)) if baseline_arr.size > 0 else 0.0
        ),
        "feature_conditioned_spread": np.float32(
            float(np.mean(spread_arr)) if spread_arr.size > 0 else 0.0
        ),
    }


def _get_tbm_geometry_outcomes(
    shared: Dict[str, Any],
    cfg: Dict[str, Any],
    tp_atr: float,
    sl_atr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (float(tp_atr), float(sl_atr))
    cache = shared.setdefault("tbm_geometry_cache", {})
    if key in cache:
        return cache[key]

    horizon = int(cfg.get("phase1_forward_horizon_bars", 12))
    close = np.asarray(shared["close"], dtype=np.float32)
    high = np.asarray(shared["high"], dtype=np.float32)
    low = np.asarray(shared["low"], dtype=np.float32)
    atr = np.asarray(shared["atr"], dtype=np.float32)
    n = close.shape[0]
    tp_first = np.zeros(n, dtype=np.int8)
    sl_first = np.zeros(n, dtype=np.int8)
    timeout = np.zeros(n, dtype=np.int8)

    for _, idxs in shared["asset_groups"].items():
        if idxs.shape[0] <= horizon + 1:
            continue
        tp_l, sl_l, to_l = tbm_outcomes_atr_nb(
            close[idxs],
            high[idxs],
            low[idxs],
            atr[idxs],
            horizon,
            float(tp_atr),
            float(sl_atr),
        )
        tp_first[idxs] = tp_l
        sl_first[idxs] = sl_l
        timeout[idxs] = to_l

    cache[key] = (tp_first, sl_first, timeout)
    return cache[key]


def _compute_tbm_economic_gain(
    shared: Dict[str, Any],
    side_mask: np.ndarray,
    mode: str,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    fee = float(cfg.get("round_trade_fee", 0.003))
    horizon = int(cfg.get("phase1_forward_horizon_bars", 12))
    close = np.asarray(shared["close"], dtype=np.float32)
    high = np.asarray(shared["high"], dtype=np.float32)
    low = np.asarray(shared["low"], dtype=np.float32)
    atr = np.asarray(shared["atr"], dtype=np.float32)
    mfe_atr = np.asarray(shared["mfe_atr"], dtype=np.float32)
    mae_atr = np.asarray(shared["mae_atr"], dtype=np.float32)
    eval_mask = (
        np.isfinite(close)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(atr)
        & np.isfinite(mfe_atr)
        & np.isfinite(mae_atr)
    )
    valid = side_mask.astype(bool) & eval_mask
    baseline_mask = eval_mask & (~side_mask.astype(bool))

    geometries = ((1.25, 0.50), (1.50, 0.60), (1.75, 0.70), (2.00, 0.90), (2.50, 1.10))
    per_geometry_metrics: List[Dict[str, Any]] = []
    geom_scores: List[float] = []
    cov_weights: List[float] = []
    cov_values: List[float] = []

    for tp_atr, sl_atr in geometries:
        tp_first, sl_first, timeout = _get_tbm_geometry_outcomes(
            shared, cfg, float(tp_atr), float(sl_atr)
        )

        tp_rate_g = float(np.mean(tp_first[valid])) if np.any(valid) else 0.0
        sl_rate_g = float(np.mean(sl_first[valid])) if np.any(valid) else 0.0
        timeout_rate_g = float(np.mean(timeout[valid])) if np.any(valid) else 1.0
        trade_rate_g = tp_rate_g + sl_rate_g
        ev_event_g = tp_rate_g * tp_atr - sl_rate_g * sl_atr - trade_rate_g * fee
        ev_trade_g = ev_event_g / max(trade_rate_g, 1e-9)
        win_rate_g = tp_rate_g / max(trade_rate_g, 1e-9)
        value_per_trade_g = ev_trade_g * win_rate_g

        base_tp = (
            float(np.mean(tp_first[baseline_mask])) if np.any(baseline_mask) else 0.0
        )
        base_sl = (
            float(np.mean(sl_first[baseline_mask])) if np.any(baseline_mask) else 0.0
        )
        base_trade = base_tp + base_sl
        base_ev_event = base_tp * tp_atr - base_sl * sl_atr - base_trade * fee
        baseline_ev_trade = base_ev_event / max(base_trade, 1e-9)
        lift_g = ev_trade_g - baseline_ev_trade

        mfe_cov_g = (
            float(np.mean(mfe_atr[valid] >= np.float32(tp_atr)))
            if np.any(valid)
            else 0.0
        )
        mae_pressure_g = (
            float(np.mean(mae_atr[valid] >= np.float32(sl_atr)))
            if np.any(valid)
            else 1.0
        )

        fold_ev: List[float] = []
        fold_lift: List[float] = []
        fold_trade: List[float] = []
        for _, va in folds:
            vv = valid[va]
            if not np.any(vv):
                continue
            tp_f = float(np.mean(tp_first[va][vv]))
            sl_f = float(np.mean(sl_first[va][vv]))
            tr_f = tp_f + sl_f
            ev_f = (tp_f * tp_atr - sl_f * sl_atr - tr_f * fee) / max(tr_f, 1e-9)
            base_v = baseline_mask[va]
            btp_f = float(np.mean(tp_first[va][base_v])) if np.any(base_v) else 0.0
            bsl_f = float(np.mean(sl_first[va][base_v])) if np.any(base_v) else 0.0
            btr_f = btp_f + bsl_f
            bev_f = (btp_f * tp_atr - bsl_f * sl_atr - btr_f * fee) / max(btr_f, 1e-9)
            fold_ev.append(ev_f)
            fold_lift.append(ev_f - bev_f)
            fold_trade.append(tr_f)

        fold_ev_arr = np.asarray(fold_ev, dtype=np.float32)
        if fold_ev_arr.size > 0:
            econ_stability_g = 0.5 * max(
                0.0,
                1.0
                - float(np.std(fold_ev_arr))
                / (abs(float(np.mean(fold_ev_arr))) + 1e-9),
            )
            econ_stability_g += 0.5 * float(np.mean(fold_ev_arr > 0.0))
        else:
            econ_stability_g = 0.0
        opportunity_adjustment_g = (
            min(1.0, np.sqrt(trade_rate_g / 0.20)) if trade_rate_g > 0 else 0.0
        )
        geometry_score_g = (
            (
                (0.35 * value_per_trade_g)
                + (0.25 * lift_g)
                + (0.15 * trade_rate_g)
                + (0.15 * max(0.0, mfe_cov_g - 0.25))
                - (0.10 * mae_pressure_g)
            )
            * opportunity_adjustment_g
            * econ_stability_g
        )

        per_geometry_metrics.append(
            {
                "tp_atr": float(tp_atr),
                "sl_atr": float(sl_atr),
                "tp_first_rate_g": tp_rate_g,
                "sl_first_rate_g": sl_rate_g,
                "timeout_rate_g": timeout_rate_g,
                "trade_opportunity_rate_g": trade_rate_g,
                "ev_net_per_trade_g": ev_trade_g,
                "lift_g": lift_g,
                "mfe_coverage_g": mfe_cov_g,
                "mae_breach_pressure_g": mae_pressure_g,
                "econ_stability_g": econ_stability_g,
                "geometry_score_g": geometry_score_g,
                "ev_net_per_trade_g_fold": fold_ev,
                "lift_g_fold": fold_lift,
                "trade_opportunity_rate_g_fold": fold_trade,
            }
        )
        geom_scores.append(float(geometry_score_g))
        cov_weights.append(max(trade_rate_g, 1e-9))
        cov_values.append(mfe_cov_g)

    top3 = np.sort(np.asarray(geom_scores, dtype=np.float32))[-3:]
    economic_gain_r = (
        0.7 * float(np.mean(top3)) + 0.3 * float(np.min(top3)) if top3.size > 0 else 0.0
    )
    aggregate_mfe_coverage = (
        float(
            np.average(
                np.asarray(cov_values, dtype=np.float32),
                weights=np.asarray(cov_weights, dtype=np.float32),
            )
        )
        if cov_values
        else 0.0
    )
    return {
        "economic_gain_r": np.float32(economic_gain_r),
        "geometry_weighted_mfe_coverage": np.float32(aggregate_mfe_coverage),
        "aggregate_mfe_coverage": np.float32(aggregate_mfe_coverage),
        "per_geometry_metrics": per_geometry_metrics,
    }


def _compute_phase4_tbm_lgbm_metrics(
    shared: Dict[str, Any],
    side_mask: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
    per_geometry_metrics: List[Dict[str, Any]],
) -> Dict[str, np.float32 | str]:
    out: Dict[str, np.float32 | str] = {
        "tbm_lgbm_auc_regime": np.float32(0.5),
        "tbm_lgbm_auc_baseline": np.float32(0.5),
        "tbm_lgbm_auc_lift_vs_baseline": np.float32(0.0),
        "tbm_lgbm_top_bucket_lift_vs_baseline": np.float32(0.0),
        "tbm_lgbm_positive_fold_fraction": np.float32(0.0),
        "tbm_lgbm_stability": np.float32(0.0),
        "tbm_lgbm_selected_geometry": "",
    }
    if not per_geometry_metrics:
        return out

    X = np.asarray(shared["learn_X"], dtype=np.float32)
    close = np.asarray(shared["close"], dtype=np.float32)
    high = np.asarray(shared["high"], dtype=np.float32)
    low = np.asarray(shared["low"], dtype=np.float32)
    atr = np.asarray(shared["atr"], dtype=np.float32)
    mfe_atr = np.asarray(shared["mfe_atr"], dtype=np.float32)
    mae_atr = np.asarray(shared["mae_atr"], dtype=np.float32)
    eval_mask = (
        np.isfinite(close)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(atr)
        & np.isfinite(mfe_atr)
        & np.isfinite(mae_atr)
    )

    timestamps = np.asarray(shared["timestamps"])
    max_subset = int(cfg.get("phase2_metric_max_samples_per_class", 25_000))
    n_splits = max(int(cfg.get("phase4_tbm_lgbm_n_splits", len(folds) or 2)), 2)
    sorted_geoms = sorted(
        per_geometry_metrics,
        key=lambda g: float(g.get("geometry_score_g", float("-inf"))),
        reverse=True,
    )

    reg_metrics: Dict[str, Any] | None = None
    non_metrics: Dict[str, Any] | None = None
    full_metrics: Dict[str, Any] | None = None
    tp_atr = 1.25
    sl_atr = 0.50
    for geom in sorted_geoms:
        tp_atr = float(geom.get("tp_atr", 1.25))
        sl_atr = float(geom.get("sl_atr", 0.50))
        tp_first, sl_first, _ = _get_tbm_geometry_outcomes(shared, cfg, tp_atr, sl_atr)
        resolved_mask = (tp_first.astype(bool) | sl_first.astype(bool)) & eval_mask
        regime = side_mask.astype(bool) & resolved_mask
        nonregime = (~side_mask.astype(bool)) & resolved_mask
        full = resolved_mask
        y = (tp_first.astype(np.float32) > 0.5).astype(np.float32)

        idx_reg = np.where(regime)[0].astype(np.int32)
        idx_non = np.where(nonregime)[0].astype(np.int32)
        idx_full = np.where(full)[0].astype(np.int32)
        reg_metrics = _lgbm_subset_cv_metrics(
            X, y, timestamps, idx_reg, n_splits=n_splits, max_subset=max_subset
        )
        non_metrics = _lgbm_subset_cv_metrics(
            X, y, timestamps, idx_non, n_splits=n_splits, max_subset=max_subset
        )
        full_metrics = _lgbm_subset_cv_metrics(
            X, y, timestamps, idx_full, n_splits=n_splits, max_subset=max_subset
        )
        reg_auc_probe = _metric_or_nan(reg_metrics["auc_mean"])
        non_auc_probe = _metric_or_nan(non_metrics["auc_mean"])
        full_auc_probe = _metric_or_nan(full_metrics["auc_mean"])
        if np.isfinite(reg_auc_probe) and (
            np.isfinite(non_auc_probe) or np.isfinite(full_auc_probe)
        ):
            break

    out["tbm_lgbm_selected_geometry"] = f"tp={tp_atr:.2f}|sl={sl_atr:.2f}"
    if reg_metrics is None or non_metrics is None or full_metrics is None:
        return out

    reg_auc_mean = _metric_or_nan(reg_metrics["auc_mean"])
    non_auc_mean = _metric_or_nan(non_metrics["auc_mean"])
    full_auc_mean = _metric_or_nan(full_metrics["auc_mean"])
    reg_lift_mean = _metric_or_nan(reg_metrics["lift_mean"])
    non_lift_mean = _metric_or_nan(non_metrics["lift_mean"])
    full_lift_mean = _metric_or_nan(full_metrics["lift_mean"])
    if not np.isfinite(reg_auc_mean):
        return out

    non_auc_cmp = non_auc_mean if np.isfinite(non_auc_mean) else float("-inf")
    full_auc_cmp = full_auc_mean if np.isfinite(full_auc_mean) else float("-inf")
    if non_auc_cmp >= full_auc_cmp:
        base_auc_mean = non_auc_mean
        base_auc_folds = np.asarray(non_metrics["auc_folds"], dtype=np.float32)
        base_lift_mean = non_lift_mean
        base_lift_folds = np.asarray(non_metrics["lift_folds"], dtype=np.float32)
    else:
        base_auc_mean = full_auc_mean
        base_auc_folds = np.asarray(full_metrics["auc_folds"], dtype=np.float32)
        base_lift_mean = full_lift_mean
        base_lift_folds = np.asarray(full_metrics["lift_folds"], dtype=np.float32)
    if not np.isfinite(base_auc_mean):
        return out

    reg_auc_folds = np.asarray(reg_metrics["auc_folds"], dtype=np.float32)
    reg_lift_folds = np.asarray(reg_metrics["lift_folds"], dtype=np.float32)
    k_auc = min(reg_auc_folds.size, base_auc_folds.size)
    if k_auc > 0:
        auc_arr = reg_auc_folds[:k_auc] - base_auc_folds[:k_auc]
    else:
        auc_arr = np.asarray([reg_auc_mean - base_auc_mean], dtype=np.float32)
    k_lift = min(reg_lift_folds.size, base_lift_folds.size)
    if k_lift > 0:
        lift_arr = reg_lift_folds[:k_lift] - base_lift_folds[:k_lift]
    elif np.isfinite(reg_lift_mean) and np.isfinite(base_lift_mean):
        lift_arr = np.asarray([reg_lift_mean - base_lift_mean], dtype=np.float32)
    else:
        lift_arr = np.asarray([], dtype=np.float32)

    stability = _stability_from_fold_deltas(auc_arr)
    out["tbm_lgbm_auc_regime"] = np.float32(reg_auc_mean)
    out["tbm_lgbm_auc_baseline"] = np.float32(base_auc_mean)
    out["tbm_lgbm_auc_lift_vs_baseline"] = np.float32(
        float(np.mean(auc_arr)) if auc_arr.size > 0 else 0.0
    )
    out["tbm_lgbm_top_bucket_lift_vs_baseline"] = np.float32(
        float(np.mean(lift_arr)) if lift_arr.size > 0 else 0.0
    )
    out["tbm_lgbm_positive_fold_fraction"] = np.float32(
        stability["positive_fold_fraction"]
        if np.isfinite(stability["positive_fold_fraction"])
        else 0.0
    )
    out["tbm_lgbm_stability"] = np.float32(
        stability["stability_score"]
        if np.isfinite(stability["stability_score"])
        else 0.0
    )
    return out


def _compute_mfe_coverage(
    shared: Dict[str, Any],
    side_mask: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    tp_atr = float(cfg.get("mfe_coverage_tp_atr", 1.25))
    mfe_atr = np.asarray(shared["mfe_atr"], dtype=np.float32)
    # fixed-threshold coverage uses only rows with full forward horizon (finite mfe_atr).
    valid = side_mask.astype(bool) & np.isfinite(mfe_atr)
    coverage = (
        float(np.mean(mfe_atr[valid] >= np.float32(tp_atr))) if np.any(valid) else 0.0
    )
    return {"fixed_tp_mfe_coverage": np.float32(coverage)}


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
        "bucket_primary_delta_fold_min": float("nan"),
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
        "incremental_information_delta_auc_fold_mean": float("nan"),
        "incremental_information_delta_auc_fold_std": float("nan"),
        "incremental_information_positive_fold_fraction": float("nan"),
        "incremental_information_positive_fold_count": float("nan"),
        "incremental_information_fold_count": float("nan"),
        "dispersion_to_edge_ratio": float("nan"),
        "edge_to_dispersion_ratio": float("nan"),
        "return_uplift": float(basic_directionality_edge),
        "primary_predictability_gain_is_nan": 1.0,
    }

    if idx_e.shape[0] < 50 or idx_ne.shape[0] < 50:
        if np.isfinite(impulse_shape_dispersion) and np.isfinite(
            basic_directionality_edge
        ):
            metrics["dispersion_to_edge_ratio"] = float(
                impulse_shape_dispersion / max(abs(basic_directionality_edge), 1e-6)
            )
            metrics["edge_to_dispersion_ratio"] = float(
                abs(basic_directionality_edge) / max(impulse_shape_dispersion, 1e-6)
            )
        metrics["return_uplift"] = float(basic_directionality_edge)
        return metrics

    max_samples_per_class = int(cfg.get("phase2_metric_max_samples_per_class", 25_000))
    classifier_n_splits = int(cfg.get("phase2_classifier_n_splits", 2))
    metric_fold_splits = int(cfg.get("phase2_metric_fold_splits", 3))
    incremental_info_n_splits = int(cfg.get("incremental_information_n_splits", 3))
    idx_e, idx_ne = _balanced_sample_indices(
        idx_e, idx_ne, max_samples_per_class, seed=123
    )

    # primary classifier
    auc_ne = _classifier_oof_auc(
        learn_X[idx_ne],
        y_primary[idx_ne],
        timestamps[idx_ne],
        n_splits=classifier_n_splits,
    )
    auc_e = _classifier_oof_auc(
        learn_X[idx_e],
        y_primary[idx_e],
        timestamps[idx_e],
        n_splits=classifier_n_splits,
    )
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
        metrics["bucket_primary_delta_fold_min"] = float(np.min(primary_delta_folds))

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
        reversal_utility = (
            -_signed_mode_return(MODE_PRICE_UP_TF, forward_returns)
        ).astype(np.float32)
    elif mode == MODE_PRICE_UP_MR:
        mae_arr = shared["mae_high"]
        mfe_arr = shared["mfe_high"]
        reversal_utility = _signed_mode_return(mode, forward_returns)
    elif mode == MODE_PRICE_DOWN_TF:
        mae_arr = shared["mae_low"]
        mfe_arr = shared["mfe_low"]
        reversal_utility = (
            -_signed_mode_return(MODE_PRICE_DOWN_TF, forward_returns)
        ).astype(np.float32)
    else:
        mae_arr = shared["mae_low"]
        mfe_arr = shared["mfe_low"]
        reversal_utility = _signed_mode_return(mode, forward_returns)

    mae_ne = _ridge_regression_oof_r2(
        learn_X[idx_ne],
        mae_arr[idx_ne],
        timestamps[idx_ne],
        clip_q=0.98,
        n_splits=classifier_n_splits,
    )
    mae_e = _ridge_regression_oof_r2(
        learn_X[idx_e],
        mae_arr[idx_e],
        timestamps[idx_e],
        clip_q=0.98,
        n_splits=classifier_n_splits,
    )
    metrics["MAE_predictability_gain"] = float(mae_e - mae_ne)
    metrics["mae_event_oos_r2"] = float(mae_e)

    mfe_ne = _ridge_regression_oof_r2(
        learn_X[idx_ne],
        mfe_arr[idx_ne],
        timestamps[idx_ne],
        clip_q=0.98,
        n_splits=classifier_n_splits,
    )
    mfe_e = _ridge_regression_oof_r2(
        learn_X[idx_e],
        mfe_arr[idx_e],
        timestamps[idx_e],
        clip_q=0.98,
        n_splits=classifier_n_splits,
    )
    metrics["MFE_predictability_gain"] = float(mfe_e - mfe_ne)
    metrics["mfe_event_oos_r2"] = float(mfe_e)

    rev_ne = _ridge_regression_oof_r2(
        learn_X[idx_ne],
        reversal_utility[idx_ne],
        timestamps[idx_ne],
        clip_q=0.98,
        n_splits=classifier_n_splits,
    )
    rev_e = _ridge_regression_oof_r2(
        learn_X[idx_e],
        reversal_utility[idx_e],
        timestamps[idx_e],
        clip_q=0.98,
        n_splits=classifier_n_splits,
    )
    metrics["reversal_utility_gain"] = float(rev_e - rev_ne)

    mae_folds = _ridge_regression_fold_r2s(
        learn_X[idx_e],
        mae_arr[idx_e],
        timestamps[idx_e],
        clip_q=0.98,
        n_splits=metric_fold_splits,
    )
    mfe_folds = _ridge_regression_fold_r2s(
        learn_X[idx_e],
        mfe_arr[idx_e],
        timestamps[idx_e],
        clip_q=0.98,
        n_splits=metric_fold_splits,
    )

    if np.isfinite(metrics["mfe_event_oos_r2"]) and (
        not np.isfinite(metrics["mae_event_oos_r2"])
        or metrics["mfe_event_oos_r2"] >= metrics["mae_event_oos_r2"]
    ):
        selected_folds = mfe_folds
        metrics["magnitude_delta_r"] = float(metrics["mfe_event_oos_r2"])
        metrics["selected_delta_metric"] = "mfe_event_oos_r2"
    else:
        selected_folds = mae_folds
        metrics["magnitude_delta_r"] = float(metrics["mae_event_oos_r2"])
        metrics["selected_delta_metric"] = "mae_event_oos_r2"

    stability = _stability_from_fold_deltas(selected_folds)
    metrics["magnitude_positive_fold_fraction"] = float(
        stability["positive_fold_fraction"]
    )
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
        regime_vals = np.array(
            [regime_preds["low"], regime_preds["normal"], regime_preds["high"]],
            dtype=np.float32,
        )
        valid_reg = regime_vals[np.isfinite(regime_vals)]
        if valid_reg.shape[0] > 0:
            regime_std = float(np.std(valid_reg))
            regime_min = float(np.min(valid_reg))
            regime_max = float(np.max(valid_reg))
        else:
            regime_std = regime_min = regime_max = 0.0

        # C. feature predictability ceiling
        simple_score = _simple_score_for_mode(mode, feature_dict, side_mask)
        valid_idx = np.where(np.isfinite(simple_score) & np.isfinite(forward_returns))[
            0
        ]
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

        rows.append(
            {
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
            }
        )

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
    global_signed_returns = _signed_mode_return(mode, forward_returns)

    ret_threshold = float(cfg.get("phase1_ret_threshold", 0.0))
    phase1_mask = _phase1_subsample_indices(shared, cfg, seed=42)
    phase1_shared = _build_phase_local_shared(shared, phase1_mask)
    candidate_grid = shared["candidate_grid"]
    candidate_registry: Dict[str, Dict[str, Any]] = {}

    # cache by geometry key
    geom_cache_phase1: Dict[str, Dict[str, Any]] = {}
    geom_cache_phase2: Dict[str, Dict[str, Any]] = {}
    global_z_cache: Dict[int, Dict[str, np.ndarray]] = {}

    phase1_rows: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Phase 1: 50% symbols + 50% history + cheap metrics + primary classifier only
    # -------------------------------------------------------------------------
    tprint(
        f"Phase 1 ({mode}): evaluating {len(candidate_grid)} candidates on 50% symbols + 50% history..."
    )

    global_target = _mode_primary_target(mode, forward_returns, ret_threshold)
    phase1_global_idx = np.where(phase1_mask)[0]
    phase1_global_target = global_target[phase1_mask]
    phase1_signed_returns = _signed_mode_return(mode, phase1_shared["forward_returns"])
    phase1_valid_fwd = np.isfinite(phase1_shared["forward_returns"])
    phase1_n_assets = max(1, len(phase1_shared["asset_groups"]))

    phase1_ratio = (
        float(np.sum(phase1_mask)) / float(phase1_mask.shape[0])
        if phase1_mask.shape[0] > 0
        else 1.0
    )
    phase1_min_total_events = max(
        10, int(cfg.get("phase1_min_total_events", 5000) * phase1_ratio)
    )

    global_to_phase1_local = np.full(forward_returns.shape[0], -1, dtype=np.int32)
    global_to_phase1_local[phase1_global_idx] = np.arange(
        phase1_global_idx.shape[0], dtype=np.int32
    )
    phase1_fold_val_locals: List[np.ndarray] = []
    for _, va in folds:
        loc = global_to_phase1_local[va]
        loc = loc[loc >= 0].astype(np.int32)
        phase1_fold_val_locals.append(loc)

    for z_hr, fam, param, d_hr in candidate_grid:
        z = int(z_hr * bph)
        duration_bars = int(d_hr * bph)
        key = CandidateKey(
            fam, int(z_hr), _safe_param_to_string(param), int(d_hr)
        ).as_str()
        candidate_registry[key] = {
            "family": fam,
            "z_hours": int(z_hr),
            "duration_hours": int(d_hr),
            "param": param,
        }

        if key not in geom_cache_phase1:
            if z not in global_z_cache:
                tprint(f"Precomputing global rolling tensors for z={z} bars...")
                global_z_cache[z] = _compute_z_cache(
                    high=shared["high"],
                    low=shared["low"],
                    close=shared["close"],
                    ret_1=shared["ret_1"],
                    vol_g=shared["vol_g"],
                    asset_groups=shared["asset_groups"],
                    z=z,
                    bph=bph,
                )
            zc_full = global_z_cache[z]
            m_high_full, m_low_full = _generate_event_masks_fast(
                family=fam,
                param_val=param,
                up_move=zc_full["up"],
                dn_move=zc_full["dn"],
                rolling_std_up=zc_full["std_up"],
                rolling_std_dn=zc_full["std_dn"],
                asset_groups=shared["asset_groups"],
                duration_bars=duration_bars,
            )
            side_mask_full = _get_side_mask(mode, m_high_full, m_low_full)

            # Subsample for Phase 1 evaluations
            side_mask = side_mask_full[phase1_mask]
            zc_local = {
                "b_up": zc_full["b_up"][phase1_mask],
                "b_dn": zc_full["b_dn"][phase1_mask],
                "s_up": zc_full["s_up"][phase1_mask],
                "s_dn": zc_full["s_dn"][phase1_mask],
                "m_up": zc_full["m_up"][phase1_mask],
                "m_dn": zc_full["m_dn"][phase1_mask],
            }

            total_events = simple_mask_count_nb(side_mask)
            if total_events < phase1_min_total_events:
                geom_cache_phase1[key] = {"rejected": True}
                continue

            active_days_frac = active_days_fraction_nb(
                side_mask, phase1_shared["day_ids"], phase1_shared["n_days"]
            )
            if active_days_frac < float(
                cfg.get("phase1_min_active_days_fraction", 0.80)
            ):
                geom_cache_phase1[key] = {"rejected": True}
                continue

            if _mode_is_up(mode):
                coh = _coherence_metrics_single_side(
                    side_mask, zc_local["b_up"], zc_local["s_up"], zc_local["m_up"]
                )
            else:
                coh = _coherence_metrics_single_side(
                    side_mask, zc_local["b_dn"], zc_local["s_dn"], zc_local["m_dn"]
                )

            distinct = _compute_regime_distinctness_single_side(
                side_mask=side_mask,
                mode=mode,
                forward_returns=phase1_shared["forward_returns"],
                mae_high=phase1_shared["mae_high"],
                mfe_high=phase1_shared["mfe_high"],
                mae_low=phase1_shared["mae_low"],
                mfe_low=phase1_shared["mfe_low"],
            )

            ev_day_mean, ev_day_std = daily_event_stats_nb(
                side_mask, phase1_shared["day_ids"], phase1_shared["n_days"]
            )
            ev_day_per_asset = float(total_events) / float(
                max(1, phase1_shared["n_days"] * phase1_n_assets)
            )

            fold_rates = []
            fold_event_counts = []
            for val_idx_local in phase1_fold_val_locals:
                if val_idx_local.shape[0] == 0:
                    continue
                fold_event_counts.append(float(np.sum(side_mask[val_idx_local])))
                fold_rates.append(
                    fold_base_rate_nb(side_mask, phase1_global_target, val_idx_local)
                )
            fold_rate_std = (
                float(np.std(np.asarray(fold_rates, dtype=np.float32)))
                if fold_rates
                else 1.0
            )
            fold_event_count_std = (
                float(np.std(np.asarray(fold_event_counts, dtype=np.float32)))
                if fold_event_counts
                else 1.0
            )

            non_event = (~side_mask) & phase1_valid_fwd
            if np.any(side_mask & phase1_valid_fwd) and np.any(non_event):
                basic_edge = float(
                    np.nanmean(phase1_signed_returns[side_mask & phase1_valid_fwd])
                    - np.nanmean(phase1_signed_returns[non_event])
                )
            else:
                basic_edge = 0.0
            dispersion_ratio = _safe_abs_ratio(
                float(coh["impulse_shape_dispersion"]), basic_edge
            )

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
                "dispersion_to_edge_ratio": float(dispersion_ratio),
            }
            geom_cache_phase1[key] = stats

        stats = geom_cache_phase1[key]
        if stats.get("rejected", False):
            continue

        phase1_rows.append(
            {
                "name": key,
                "family": fam,
                "z_hours": z_hr,
                "param": _safe_param_to_string(param),
                "duration_hours": d_hr,
                **{k: v for k, v in stats.items() if k not in {"rejected"}},
            }
        )

    if not phase1_rows:
        return {"status": "failed", "reason": f"no_phase1_candidates_{mode}"}

    df1 = pd.DataFrame(phase1_rows)
    disp_edge_z = _zscore_np(df1["dispersion_to_edge_ratio"].values.astype(np.float32))
    disp_edge_z[np.isnan(disp_edge_z)] = 3.0  # Assign heavy penalty to missing/infinite dispersion

    df1["phase1_proxy_score"] = (
        0.20 * _zscore_np(df1["active_days_fraction"].values)
        + 0.15 * _zscore_np(df1["regime_distinctness_score"].values)
        + 0.15 * _zscore_np(np.log1p(df1["total_events"].values.astype(np.float32)))
        + 0.30 * _zscore_np(df1["basic_directionality_edge_event_vs_non_event"].values)
        - 0.30 * disp_edge_z
        - 0.15 * _zscore_np(df1["fold_continuation_rate_std"].values)
        - 0.10 * _zscore_np(df1["fold_event_count_std"].values)
    )

    _log_stage_snapshot(
        mode,
        "Phase 1",
        df1,
        "phase1_proxy_score",
        [
            "name",
            "phase1_proxy_score",
            "basic_directionality_edge_event_vs_non_event",
            "dispersion_to_edge_ratio",
            "total_events",
            "active_days_fraction",
        ],
    )

    # -------------------------------------------------------------------------
    # Stage A/B/C Diversity Filter (Phase 1)
    # -------------------------------------------------------------------------
    df1 = df1.sort_values("phase1_proxy_score", ascending=False)

    # Extract feature base and family for group logic
    df1["feature_base"] = df1["name"].apply(lambda x: candidate_registry[x].get("feature_base", x))
    df1["family"] = df1["name"].apply(lambda x: candidate_registry[x].get("family", "unknown"))

    # 1. Top 1 config per feature
    df1 = df1.drop_duplicates(subset=["feature_base"], keep="first")

    # 2. Keep top features per family (Stage 1/2 rule: at least 2 stable per family)
    # We will just select top 2 per family, then pad with global tops if needed.
    top_per_fam = df1.groupby("family").head(2)

    # Keep global top K as well
    top_k_global = df1.head(int(cfg.get("top_k_for_learnability", 8)))

    # Union and deduplicate
    df1 = pd.concat([top_per_fam, top_k_global]).drop_duplicates(subset=["name"]).sort_values("phase1_proxy_score", ascending=False).copy()


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
            zc = global_z_cache[z]
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

            active_days_frac = active_days_fraction_nb(side_mask, day_ids, n_days)
            if active_days_frac < float(
                cfg.get("phase2_min_active_days_fraction", 0.80)
            ):
                geom_cache_phase2[key] = {"rejected": True}
                continue

            if _mode_is_up(mode):
                coh = _coherence_metrics_single_side(
                    side_mask, zc["b_up"], zc["s_up"], zc["m_up"]
                )
            else:
                coh = _coherence_metrics_single_side(
                    side_mask, zc["b_dn"], zc["s_dn"], zc["m_dn"]
                )

            distinct = _compute_regime_distinctness_single_side(
                side_mask=side_mask,
                mode=mode,
                forward_returns=forward_returns,
                mae_high=shared["mae_high"],
                mfe_high=shared["mfe_high"],
                mae_low=shared["mae_low"],
                mfe_low=shared["mfe_low"],
            )

            ev_day_mean, ev_day_std = daily_event_stats_nb(side_mask, day_ids, n_days)
            ev_day_per_asset = float(total_events) / float(
                max(1, n_days * phase2_n_assets)
            )
            side_vol_exp = zc["v_exp"][side_mask & np.isfinite(zc["v_exp"])]
            post_event_vol_dispersion = (
                float(np.std(side_vol_exp)) if side_vol_exp.shape[0] > 1 else 0.0
            )

            fold_rates = [
                fold_base_rate_nb(side_mask, global_target, va) for _, va in folds
            ]
            fold_event_counts = [float(np.sum(side_mask[va])) for _, va in folds]
            fold_rate_std = (
                float(np.std(np.asarray(fold_rates, dtype=np.float32)))
                if fold_rates
                else 1.0
            )
            fold_continuation_rate_std = fold_rate_std
            fold_event_count_std = (
                float(np.std(np.asarray(fold_event_counts, dtype=np.float32)))
                if fold_event_counts
                else 1.0
            )

            valid_fwd_p2 = np.isfinite(global_signed_returns)
            non_event = (~side_mask) & valid_fwd_p2
            if np.any(side_mask & valid_fwd_p2) and np.any(non_event):
                basic_edge = float(
                    np.nanmean(global_signed_returns[side_mask & valid_fwd_p2])
                    - np.nanmean(global_signed_returns[non_event])
                )
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
            legacy_metrics = _compute_legacy_conditional_learnability(
                mode, side_mask, shared, cfg
            )

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

        phase2_rows.append(
            {
                "name": key,
                "family": fam,
                "z_hours": z_hr,
                "param": _safe_param_to_string(param),
                "duration_hours": d_hr,
                "conditioner_mode": "none",
                **{k: v for k, v in stats.items() if k not in {"rejected"}},
            }
        )

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
    df2["predictability_gain"] = df2[
        [primary_col, "MAE_predictability_gain", "MFE_predictability_gain"]
    ].max(axis=1).astype(np.float32)
    df2["delta_r_raw"] = df2["return_uplift"].astype(np.float32)
    df2["delta_r_fallback"] = (
        0.5 * df2["incremental_information_delta_auc"].astype(np.float32)
    ).astype(np.float32)

    df2["delta_r"] = df2["delta_r_raw"].astype(np.float32)

    df2["selected_delta_metric"] = "return_uplift"
    selected_fold_mean = df2["bucket_primary_delta_fold_mean"].astype(np.float32).values
    selected_fold_std = df2["bucket_primary_delta_fold_std"].astype(np.float32).values
    df2["delta_r_fold_mean"] = selected_fold_mean.astype(np.float32)
    df2["delta_r_fold_std"] = selected_fold_std.astype(np.float32)
    df2["positive_fold_fraction_r"] = df2[
        "incremental_information_positive_fold_fraction"
    ].astype(np.float32)
    df2["S_r"] = (
        0.5
        * np.maximum(
            0.0,
            1.0
            - (
                np.nan_to_num(df2["delta_r_fold_std"].values, nan=np.inf)
                / (
                    np.abs(np.nan_to_num(df2["delta_r_fold_mean"].values, nan=0.0))
                    + 1e-9
                )
            ),
        )
        + 0.5
        * np.nan_to_num(
            df2["incremental_information_positive_fold_fraction"].values, nan=0.0
        )
    ).astype(np.float32)
    df2["delta_r_shrunk"] = (
        df2["delta_r"].astype(np.float32).values
        * (
            df2["N_r"].astype(np.float32).values
            / (df2["N_r"].astype(np.float32).values + 500.0)
        )
    ).astype(np.float32)
    df2["uplift_anchor"] = np.maximum(
        df2["delta_r_shrunk"].astype(np.float32).values, 0.0
    ).astype(np.float32)
    df2["primary_multiplier"] = (
        1.0
        + np.tanh(
            10.0 * df2["bucket_primary_predictability_gain"].astype(np.float32).values
        )
    ).astype(np.float32)
    df2["worst_fold_multiplier"] = (
        1.0
        + 0.5
        * np.tanh(
            10.0
            * np.nan_to_num(
                df2["bucket_primary_delta_fold_min"].astype(np.float32).values,
                nan=0.0,
            )
        )
    ).astype(np.float32)
    df2["noise_penalty"] = (
        1.0
        + np.log1p(
            np.maximum(
                np.nan_to_num(
                    df2["dispersion_to_edge_ratio"].astype(np.float32).values,
                    nan=100.0,
                ),
                0.0,
            )
        )
    ).astype(np.float32)
    primary_sign = np.sign(
        np.nan_to_num(
            df2["bucket_primary_predictability_gain"].astype(np.float32).values, nan=0.0
        )
    ).astype(np.float32)
    uplift_sign = np.sign(np.nan_to_num(df2["delta_r_raw"].astype(np.float32).values, nan=0.0)).astype(
        np.float32
    )
    df2["disagreement_penalty"] = np.where(
        (primary_sign != 0.0) & (uplift_sign != 0.0) & (primary_sign != uplift_sign),
        np.float32(0.65),
        np.float32(1.0),
    ).astype(np.float32)

    df2["learnability_support"] = np.maximum(
        df2["incremental_information_delta_auc"].astype(np.float32).values, 0.0
    ).astype(np.float32)
    df2["effective_edge"] = (
        df2["uplift_anchor"].astype(np.float32).values
        * np.maximum(df2["S_r"].astype(np.float32).values, 0.0)
        * df2["primary_multiplier"].astype(np.float32).values
        * df2["worst_fold_multiplier"].astype(np.float32).values
        * df2["disagreement_penalty"].astype(np.float32).values
    ).astype(np.float32)
    df2["score_r"] = (
        df2["effective_edge"].astype(np.float32).values
        * (1.0 + 25.0 * df2["learnability_support"].astype(np.float32).values)
        / np.maximum(df2["noise_penalty"].astype(np.float32).values, 1e-6)
    ).astype(np.float32)
    df2["score_ml"] = df2["score_r"].astype(np.float32)

    _log_stage_snapshot(
        mode,
        "Phase 2",
        df2,
        "score_r",
        [
            "name",
            "score_r",
            "effective_edge",
            "learnability_support",
            "noise_penalty",
            "delta_r_raw",
            "incremental_information_delta_auc",
            "dispersion_to_edge_ratio",
            "disagreement_penalty",
        ],
    )


    # -------------------------------------------------------------------------
    # Phase 2.5: Ridge regime attribution
    # -------------------------------------------------------------------------
    tprint(f"Phase 2.5 ({mode}): Ridge regime attribution for top {len(df2)} candidates...")

    full_df_dict = {
        "timestamp": shared["timestamps"],
        "high": shared["high"],
        "low": shared["low"],
        "close": shared["close"],
    }
    if "open" in shared:
        full_df_dict["open"] = shared["open"]
    if "volume" in shared:
        full_df_dict["volume"] = shared["volume"]

    full_df = pd.DataFrame(full_df_dict)
    regime_features_df = build_regime_features(full_df)

    # Identify which features are binary vs continuous
    feature_types = {}
    for c in RIDGE_FEATURE_COLS:
        if c in regime_features_df.columns:
            u_vals = regime_features_df[c].dropna().unique()
            if len(u_vals) <= 2 and set(u_vals).issubset({0.0, 1.0, 0, 1}):
                feature_types[c] = "binary"
            else:
                feature_types[c] = "continuous"

    dynamic_conditioners: Dict[str, List[Dict[str, Any]]] = {}

    for _, row in df2.iterrows():
        base_name = str(row["name"])
        reg = candidate_registry[base_name]
        z = int(int(reg["z_hours"]) * bph)
        duration_bars = int(int(reg["duration_hours"]) * bph)
        if z not in global_z_cache:
            global_z_cache[z] = _compute_z_cache(
                high=shared["high"],
                low=shared["low"],
                close=shared["close"],
                ret_1=shared["ret_1"],
                vol_g=shared["vol_g"],
                asset_groups=shared["asset_groups"],
                z=z,
                bph=bph,
                volume=shared.get("volume", None),
            )
        zc = global_z_cache[z]
        m_high, m_low = _generate_event_masks_fast(
            candidate=reg,
            zc=zc,
            asset_groups=shared["asset_groups"],
        )
        side_mask = _get_side_mask(mode, m_high, m_low)

        fwd_2h_bars = int(2 * bph)
        if "close" in full_df:
            fwd_ret = full_df["close"].shift(-fwd_2h_bars) / full_df["close"] - 1.0
        else:
            fwd_ret = pd.Series(np.zeros(len(full_df)))

        ridge_df = regime_features_df.copy()
        ridge_df["event_mask"] = side_mask.astype(int)
        ridge_df["target_fwd_return"] = fwd_ret

        valid_feature_cols = [c for c in RIDGE_FEATURE_COLS if c in ridge_df.columns]

        res = fit_ridge_regime_scan(
            ridge_df,
            valid_feature_cols,
            "event_mask",
            "target_fwd_return",
            n_splits=max(2, len(folds))
        )

        cond_features = []
        if res is not None:
            seeds = res.get("phase3_conditioner_seeds", [])
            # We already have seeds outputted from the updated function
            for seed in seeds:
                cond_features.append({
                    "feature": seed.feature,
                    "coef": seed.coefficient,
                    "abs_signed_importance": seed.abs_signed_importance,
                    "type": seed.feature_type,
                    "thresholds": seed.thresholds
                })

        dynamic_conditioners[base_name] = cond_features

    feature_gain_vals: List[np.float32] = []
    feature_pos_vals: List[np.float32] = []
    top_feature_lifts_vals: List[str] = []
    cond_gain_vals: List[np.float32] = []
    cond_pos_vals: List[np.float32] = []
    cond_regime_r2_vals: List[np.float32] = []
    cond_base_r2_vals: List[np.float32] = []
    cond_spread_vals: List[np.float32] = []
    econ_vals: List[np.float32] = []
    agg_mfe_cov_vals: List[np.float32] = []
    fixed_cov_vals: List[np.float32] = []
    per_geom_vals: List[Any] = []
    tbm_auc_regime_vals: List[np.float32] = []
    tbm_auc_base_vals: List[np.float32] = []
    tbm_auc_lift_vals: List[np.float32] = []
    tbm_top_lift_vals: List[np.float32] = []
    tbm_pos_vals: List[np.float32] = []
    tbm_stability_vals: List[np.float32] = []
    tbm_geom_name_vals: List[str] = []
    for _, row in df2.iterrows():
        reg = candidate_registry[str(row["name"])]
        z = int(int(reg["z_hours"]) * bph)
        duration_bars = int(int(reg["duration_hours"]) * bph)
        if z not in global_z_cache:
            global_z_cache[z] = _compute_z_cache(
                high=shared["high"],
                low=shared["low"],
                close=shared["close"],
                ret_1=shared["ret_1"],
                vol_g=shared["vol_g"],
                asset_groups=shared["asset_groups"],
                z=z,
                bph=bph,
                volume=shared.get("volume", None),
            )
        zc = global_z_cache[z]
        m_high, m_low = _generate_event_masks_fast(
            candidate=reg,
            zc=zc,
            asset_groups=shared["asset_groups"],
        )
        side_mask = _get_side_mask(mode, m_high, m_low)
        feat_metrics = _compute_phase3_feature_learnability(
            shared, feature_dict, side_mask, mode, folds, cfg
        )
        cond_metrics = _compute_conditional_predictability_metrics(
            shared, side_mask, mode, folds, cfg
        )
        econ_metrics = _compute_tbm_economic_gain(shared, side_mask, mode, folds, cfg)
        mfe_metrics = _compute_mfe_coverage(shared, side_mask, cfg)
        tbm_lgbm_metrics = _compute_phase4_tbm_lgbm_metrics(
            shared,
            side_mask,
            folds,
            cfg,
            econ_metrics["per_geometry_metrics"],
        )

        feature_gain_vals.append(np.float32(feat_metrics["feature_learnability_gain"]))
        feature_pos_vals.append(
            np.float32(feat_metrics["feature_positive_fold_fraction"])
        )
        top_feature_lifts_vals.append(str(feat_metrics["top_feature_lifts"]))
        cond_gain_vals.append(
            np.float32(cond_metrics["conditional_predictability_gain"])
        )
        cond_pos_vals.append(
            np.float32(cond_metrics["conditional_predictability_positive_fold_fraction"])
        )
        cond_regime_r2_vals.append(
            np.float32(cond_metrics["conditional_predictability_regime_r2"])
        )
        cond_base_r2_vals.append(
            np.float32(cond_metrics["conditional_predictability_baseline_r2"])
        )
        cond_spread_vals.append(np.float32(cond_metrics["feature_conditioned_spread"]))
        econ_vals.append(np.float32(econ_metrics["economic_gain_r"]))
        agg_mfe_cov_vals.append(
            np.float32(econ_metrics["geometry_weighted_mfe_coverage"])
        )
        fixed_cov_vals.append(np.float32(mfe_metrics["fixed_tp_mfe_coverage"]))
        per_geom_vals.append(econ_metrics["per_geometry_metrics"])
        tbm_auc_regime_vals.append(
            np.float32(tbm_lgbm_metrics["tbm_lgbm_auc_regime"])
        )
        tbm_auc_base_vals.append(
            np.float32(tbm_lgbm_metrics["tbm_lgbm_auc_baseline"])
        )
        tbm_auc_lift_vals.append(
            np.float32(tbm_lgbm_metrics["tbm_lgbm_auc_lift_vs_baseline"])
        )
        tbm_top_lift_vals.append(
            np.float32(tbm_lgbm_metrics["tbm_lgbm_top_bucket_lift_vs_baseline"])
        )
        tbm_pos_vals.append(
            np.float32(tbm_lgbm_metrics["tbm_lgbm_positive_fold_fraction"])
        )
        tbm_stability_vals.append(
            np.float32(tbm_lgbm_metrics["tbm_lgbm_stability"])
        )
        tbm_geom_name_vals.append(str(tbm_lgbm_metrics["tbm_lgbm_selected_geometry"]))

    df2["feature_learnability_gain"] = np.asarray(feature_gain_vals, dtype=np.float32)
    df2["feature_positive_fold_fraction"] = np.asarray(
        feature_pos_vals, dtype=np.float32
    )
    df2["top_feature_lifts"] = top_feature_lifts_vals
    df2["conditional_predictability_gain"] = np.asarray(
        cond_gain_vals, dtype=np.float32
    )
    df2["conditional_predictability_positive_fold_fraction"] = np.asarray(
        cond_pos_vals, dtype=np.float32
    )
    df2["conditional_predictability_regime_r2"] = np.asarray(
        cond_regime_r2_vals, dtype=np.float32
    )
    df2["conditional_predictability_baseline_r2"] = np.asarray(
        cond_base_r2_vals, dtype=np.float32
    )
    df2["feature_conditioned_spread"] = np.asarray(
        cond_spread_vals, dtype=np.float32
    )
    df2["economic_gain_r"] = np.asarray(econ_vals, dtype=np.float32)
    df2["geometry_weighted_mfe_coverage"] = np.asarray(
        agg_mfe_cov_vals, dtype=np.float32
    )
    df2["fixed_tp_mfe_coverage"] = np.asarray(fixed_cov_vals, dtype=np.float32)
    df2["aggregate_mfe_coverage"] = df2["geometry_weighted_mfe_coverage"].astype(
        np.float32
    )
    df2["per_geometry_metrics"] = per_geom_vals
    df2["tbm_lgbm_auc_regime"] = np.asarray(tbm_auc_regime_vals, dtype=np.float32)
    df2["tbm_lgbm_auc_baseline"] = np.asarray(tbm_auc_base_vals, dtype=np.float32)
    df2["tbm_lgbm_auc_lift_vs_baseline"] = np.asarray(
        tbm_auc_lift_vals, dtype=np.float32
    )
    df2["tbm_lgbm_top_bucket_lift_vs_baseline"] = np.asarray(
        tbm_top_lift_vals, dtype=np.float32
    )
    df2["tbm_lgbm_positive_fold_fraction"] = np.asarray(
        tbm_pos_vals, dtype=np.float32
    )
    df2["tbm_lgbm_stability"] = np.asarray(tbm_stability_vals, dtype=np.float32)
    df2["tbm_lgbm_selected_geometry"] = tbm_geom_name_vals
    min_mfe_cov = float(cfg.get("mask_opt_min_mfe_coverage", 0.02))
    _log_stage_snapshot(
        mode,
        "Phase 3",
        df2,
        "conditional_predictability_gain",
        [
            "name",
            "conditional_predictability_gain",
            "conditional_predictability_positive_fold_fraction",
            "conditional_predictability_regime_r2",
            "conditional_predictability_baseline_r2",
            "feature_conditioned_spread",
            "feature_learnability_gain",
            "feature_positive_fold_fraction",
            "score_r",
            "delta_r_raw",
        ],
    )
    _log_stage_snapshot(
        mode,
        "Stage 4 Coverage",
        df2,
        "aggregate_mfe_coverage",
        [
            "name",
            "aggregate_mfe_coverage",
            "economic_gain_r",
            "score_r",
            "delta_r_raw",
        ],
    )
    df2["coverage_multiplier"] = (
        0.25
        + 0.75
        * np.clip(
            df2["aggregate_mfe_coverage"].astype(np.float32).values
            / max(min_mfe_cov, 1e-6),
            0.0,
            1.0,
        )
    ).astype(np.float32)
    df2["predictability_anchor"] = np.maximum(
        df2["conditional_predictability_gain"].astype(np.float32).values, 0.0
    ).astype(np.float32)
    df2["predictability_positive_multiplier"] = (
        0.75
        + 0.25
        * df2["conditional_predictability_positive_fold_fraction"].astype(np.float32).values
    ).astype(np.float32)
    df2["spread_multiplier"] = (
        1.0
        + np.tanh(10.0 * df2["feature_conditioned_spread"].astype(np.float32).values)
    ).astype(np.float32)
    df2["difference_prior"] = (
        0.85 + 0.15 * np.clip(df2["score_r"].astype(np.float32).values, 0.0, None)
    ).astype(np.float32)
    df2["score_ml"] = (
        df2["score_r"].astype(np.float32).values
        * (
            1.0
            + 5.0 * df2["predictability_anchor"].astype(np.float32).values
        )
        * df2["predictability_positive_multiplier"].astype(np.float32).values
        * df2["spread_multiplier"].astype(np.float32).values
        * df2["difference_prior"].astype(np.float32).values
    ).astype(np.float32)
    df2["tbm_auc_support"] = np.maximum(
        df2["tbm_lgbm_auc_lift_vs_baseline"].astype(np.float32).values, 0.0
    ).astype(np.float32)
    df2["tbm_lift_support"] = np.maximum(
        df2["tbm_lgbm_top_bucket_lift_vs_baseline"].astype(np.float32).values,
        0.0,
    ).astype(np.float32)
    df2["score_ml_trading"] = (
        df2["score_ml"].astype(np.float32).values
        * (1.0 + 2.0 * np.maximum(df2["feature_learnability_gain"].astype(np.float32).values, 0.0))
        * (1.0 + 25.0 * df2["tbm_auc_support"].astype(np.float32).values)
        * (1.0 + 10.0 * df2["tbm_lift_support"].astype(np.float32).values)
        * (
            0.50
            + 0.50
            * np.maximum(df2["tbm_lgbm_stability"].astype(np.float32).values, 0.0)
        )
        * (
            0.50
            + 0.50
            * np.maximum(
                df2["tbm_lgbm_positive_fold_fraction"].astype(np.float32).values, 0.0
            )
        )
        * df2["coverage_multiplier"].astype(np.float32).values
    ).astype(np.float32)
    df2["shortlist_score"] = df2["score_ml_trading"].astype(np.float32)
    df2["decision"] = "ranked"
    df2["regime_id"] = df2["name"].astype(str)
    df2["regime_definition"] = df2["name"].astype(str)
    df2["rationale"] = df2.apply(_build_regime_rationale, axis=1)

    df2 = df2.sort_values(
        [
            "score_ml_trading",
            "economic_gain_r",
            "feature_learnability_gain",
            "delta_r",
            "total_events",
        ],
        ascending=[False, False, False, False, False],
    )

    df2["feature_base"] = df2["name"].apply(lambda x: candidate_registry[str(x)].get("feature_base", str(x)))
    df2["family"] = df2["name"].apply(lambda x: candidate_registry[str(x)].get("family", "unknown"))

    # Stage 2.5/3 diversity: global top + at least 1 stable per family, max 3 per family
    df2 = df2.drop_duplicates(subset=["feature_base"], keep="first")

    stage3_max = int(cfg.get("stage3_max_candidates", 10))
    shortlist_max = int(cfg.get("shortlist_max_candidates", stage3_max))

    # Keep at least 1 per family
    top_1_fam = df2.groupby("family").head(1)

    # Pad with global top, but enforce max 3 per family
    df_short_list = []
    fam_counts = {fam: 0 for fam in df2["family"].unique()}

    # First add the guaranteed 1 per family
    for _, row in top_1_fam.iterrows():
        df_short_list.append(row)
        fam_counts[row["family"]] += 1

    # Then add from global list until we hit shortlist_max
    for _, row in df2.iterrows():
        if len(df_short_list) >= shortlist_max:
            break
        if fam_counts[row["family"]] < 3:
            # Check if not already added
            if not any(r["name"] == row["name"] for r in df_short_list):
                df_short_list.append(row)
                fam_counts[row["family"]] += 1

    if not df_short_list:
        return {
            "status": "failed",
            "reason": f"no_shortlist_candidates_{mode}",
            "layer0_candidate_table_": df2,
        }

    df_short = pd.DataFrame(df_short_list)
    # Ensure at least 10 if viable
    # If df_short < 10, we could add more, but we are capped by shortlist_max.

    df_short["tier"] = 0
    df_short["conditioner_mode"] = "none"

    candidate_masks: Dict[str, Dict[str, np.ndarray]] = {}
    for _, row in df_short.iterrows():
        name = row["name"]
        reg = candidate_registry[name]
        z = int(int(reg["z_hours"]) * bph)
        duration_bars = int(int(reg["duration_hours"]) * bph)
        if z not in global_z_cache:
            tprint(f"Precomputing global rolling tensors for z={z} bars...")
            global_z_cache[z] = _compute_z_cache(
                high=shared["high"],
                low=shared["low"],
                close=shared["close"],
                ret_1=shared["ret_1"],
                vol_g=shared["vol_g"],
                asset_groups=shared["asset_groups"],
                z=z,
                bph=bph,
                volume=shared.get("volume", None),
            )
        zc = global_z_cache[z]
        m_high, m_low = _generate_event_masks_fast(
            candidate=reg,
            zc=zc,
            asset_groups=shared["asset_groups"],
        )
        candidate_masks[name] = {"m_high": m_high, "m_low": m_low}

    cond_rows: List[pd.Series] = []
    if bool(cfg.get("enable_secondary_conditioners", True)):
        # Configurable limits
        min_events = int(cfg.get("phase3_min_conditioned_event_count", 2000))
        min_fraction = float(cfg.get("phase3_min_event_fraction_of_base", 0.10))
        tier2_min_fraction = float(cfg.get("phase3_tier2_min_event_fraction", 0.05))
        max_singles = int(cfg.get("phase3_max_single_candidates_per_base", 4))
        max_pairs = int(cfg.get("phase3_max_pair_candidates", 10))

        for _, row in df_short.iterrows():
            cand_name = str(row["name"])
            reg = candidate_registry[cand_name]
            z = int(int(reg["z_hours"]) * bph)
            zc = global_z_cache[z]
            base_masks = candidate_masks[cand_name]
            base_side_mask = _get_side_mask(mode, base_masks["m_high"], base_masks["m_low"])
            base_event_count = int(np.sum(base_side_mask))

            # ---------------------------------------------------------
            # 3A. Generate Single-Regime Candidates (Tier-1)
            # ---------------------------------------------------------
            tier1_candidates = []
            top_vars = dynamic_conditioners.get(cand_name, [])

            for var_info in top_vars:
                var_name = var_info["feature"]
                coef = var_info["coef"]
                v_type = var_info["type"]
                family = var_info.get("family", "unknown")

                if var_name not in regime_features_df.columns:
                    continue

                feature_vals = regime_features_df[var_name].values
                valid_mask = np.isfinite(feature_vals)
                active_valid = base_side_mask & valid_mask
                if np.sum(active_valid) < 50:
                    continue

                if v_type == "binary":
                    target_val = 1 if coef > 0 else 0
                    cond_mask = valid_mask & (feature_vals == target_val)
                    tier1_candidates.append({
                        "name": f"{cand_name}_{var_name}_is_{target_val}",
                        "desc": f"{var_name} == {target_val}",
                        "mask": cond_mask,
                        "features": [var_name],
                        "families": [family]
                    })
                else:
                    direction = "gt" if coef > 0 else "lt"
                    thresholds_dict = var_info.get("thresholds")
                    if not thresholds_dict:
                        continue

                    quantiles_to_check = ["q50", "q60", "q70", "q80"] if coef > 0 else ["q50", "q40", "q30", "q20"]
                    for q_key in quantiles_to_check:
                        if q_key in thresholds_dict:
                            threshold = thresholds_dict[q_key]
                            if direction == "gt":
                                cond_mask = valid_mask & (feature_vals > threshold)
                                desc = f"{var_name} > {q_key}"
                            else:
                                cond_mask = valid_mask & (feature_vals < threshold)
                                desc = f"{var_name} < {q_key}"

                            tier1_candidates.append({
                                "name": f"{cand_name}_{var_name}_{desc.replace(' ', '').replace('>', 'gt').replace('<', 'lt')}",
                                "desc": desc,
                                "mask": cond_mask,
                                "features": [var_name],
                                "families": [family]
                            })

                    # Add middle band 30-70
                    if "q30" in thresholds_dict and "q70" in thresholds_dict:
                        cond_mask = valid_mask & (feature_vals > thresholds_dict["q30"]) & (feature_vals < thresholds_dict["q70"])
                        desc = f"{var_name} > q30 AND {var_name} < q70"
                        tier1_candidates.append({
                            "name": f"{cand_name}_{var_name}_gt_q30_lt_q70",
                            "desc": desc,
                            "mask": cond_mask,
                            "features": [var_name],
                            "families": [family]
                        })

                    # Add broad exclusion rule 20-80
                    if "q20" in thresholds_dict and "q80" in thresholds_dict:
                        cond_mask = valid_mask & (feature_vals > thresholds_dict["q20"]) & (feature_vals < thresholds_dict["q80"])
                        desc = f"{var_name} > q20 AND {var_name} < q80"
                        tier1_candidates.append({
                            "name": f"{cand_name}_{var_name}_gt_q20_lt_q80",
                            "desc": desc,
                            "mask": cond_mask,
                            "features": [var_name],
                            "families": [family]
                        })

            # Base Evaluation Closure
            def eval_candidate(c_info, tier, parent_res=None):
                new_side_mask = base_side_mask & c_info["mask"]
                tot_events = int(np.sum(new_side_mask))

                req_fraction = min_fraction if tier == 1 else tier2_min_fraction
                if tot_events < min_events or (tot_events / base_event_count) < req_fraction:
                    return None

                coh = (
                    _coherence_metrics_single_side(new_side_mask, zc["b_up"], zc["s_up"], zc["m_up"])
                    if _mode_is_up(mode)
                    else _coherence_metrics_single_side(new_side_mask, zc["b_dn"], zc["s_dn"], zc["m_dn"])
                )

                valid_fwd_new = np.isfinite(global_signed_returns)
                non_event_new = (~new_side_mask) & valid_fwd_new
                basic_edge_new = (
                    float(np.nanmean(global_signed_returns[new_side_mask & valid_fwd_new]) - np.nanmean(global_signed_returns[non_event_new]))
                    if np.any(new_side_mask & valid_fwd_new) and np.any(non_event_new)
                    else 0.0
                )

                new_metrics = _compute_full_metrics_for_candidate(
                    mode,
                    new_side_mask,
                    shared,
                    feature_dict,
                    cfg,
                    float(coh["impulse_shape_dispersion"]),
                    float(basic_edge_new),
                )

                econ_metrics = _compute_tbm_economic_gain(shared, new_side_mask, mode, folds, cfg)
                mfe_metrics = _compute_mfe_coverage(shared, new_side_mask, cfg)
                new_econ = _metric_or_nan(econ_metrics.get("economic_gain_r"))
                new_mfe = _metric_or_nan(mfe_metrics.get("fixed_tp_mfe_coverage"))

                # In order to do base comparison, we need base_econ. If row doesn't have it, we must compute it.
                if "economic_gain_r" not in row:
                    base_econ_metrics = _compute_tbm_economic_gain(shared, base_side_mask, mode, folds, cfg)
                    base_econ = _metric_or_nan(base_econ_metrics.get("economic_gain_r"))
                    row["economic_gain_r"] = base_econ

                    base_mfe_metrics = _compute_mfe_coverage(shared, base_side_mask, cfg)
                    base_mfe = _metric_or_nan(base_mfe_metrics.get("fixed_tp_mfe_coverage"))
                    row["aggregate_mfe_coverage"] = base_mfe
                else:
                    base_econ = _metric_or_nan(row.get("economic_gain_r"))
                    base_mfe = _metric_or_nan(row.get("aggregate_mfe_coverage"))

                improves_econ = (new_econ > base_econ * 1.05)
                improves_mfe = (new_mfe > base_mfe * 1.05)

                # Check net regime value
                best_geom = econ_metrics.get("per_geometry_metrics", [{}])[0]
                labels_ER = best_geom.get("labels", np.array([]))

                base_best_geom = _compute_tbm_economic_gain(shared, base_side_mask, mode, folds, cfg).get("per_geometry_metrics", [{}])[0]
                labels_E = base_best_geom.get("labels", np.array([]))

                auc_ER = quick_ridge_auc(regime_features_df, labels_ER, new_side_mask, folds)
                auc_E = quick_ridge_auc(regime_features_df, labels_E, base_side_mask, folds)

                fwd_ret_ER = global_signed_returns[new_side_mask & valid_fwd_new]
                fwd_ret_E = global_signed_returns[base_side_mask & valid_fwd_new]

                nrv_score, nrv_diags = compute_net_regime_value(
                    returns_E=fwd_ret_E,
                    returns_ER=fwd_ret_ER,
                    delta_r_folds_E=np.array([float(np.nanmean(global_signed_returns[(base_side_mask & valid_fwd_new) & va])) for _, va in folds]),
                    delta_r_folds_ER=np.array([float(np.nanmean(global_signed_returns[(new_side_mask & valid_fwd_new) & va])) for _, va in folds]),
                    labels_E=labels_E[base_side_mask] if len(labels_E) == len(base_side_mask) else np.array([]),
                    labels_ER=labels_ER[new_side_mask] if len(labels_ER) == len(new_side_mask) else np.array([]),
                    auc_E=auc_E,
                    auc_ER=auc_ER,
                )

                new_metrics["net_regime_value"] = nrv_score

                # Stronger acceptance rules based on prompt
                der_ratio = nrv_diags["DER_ratio"]
                sr_ratio = nrv_diags["S_r_ratio"]

                # Check for deterioration
                is_stability_worse = (sr_ratio < 0.90)
                is_dispersion_worse = (der_ratio < 0.90)

                if tier == 1:
                    if not (improves_econ or improves_mfe or nrv_score > 1.05):
                        return None
                    if is_stability_worse or is_dispersion_worse:
                        return None

                if tier == 2:
                    # Compare against BEST single parent if provided
                    if parent_res is not None:
                        parent_econ = _metric_or_nan(parent_res.get("economic_gain_r"))
                        parent_mfe = _metric_or_nan(parent_res.get("aggregate_mfe_coverage"))
                        parent_nrv = _metric_or_nan(parent_res.get("net_regime_value"))

                        if not (new_econ > parent_econ * 1.05 or new_mfe > parent_mfe * 1.05 or nrv_score > parent_nrv * 1.05):
                            return None
                    else:
                        if not (new_econ > base_econ * 1.1 or new_mfe > base_mfe * 1.1 or nrv_score > 1.1):
                            return None
                    if is_stability_worse or is_dispersion_worse:
                        return None

                # Build row
                new_row = row.copy()
                new_row["name"] = c_info["name"]
                new_row["conditioner_mode"] = c_info["desc"]
                new_row["tier"] = tier
                new_row["total_events"] = tot_events
                new_row["impulse_shape_dispersion"] = float(coh["impulse_shape_dispersion"])

                for k, v in new_metrics.items():
                    new_row[k] = v

                new_row["delta_r_raw"] = float(basic_edge_new)
                new_row["delta_r_fallback"] = (
                    float(0.5 * new_row["incremental_information_delta_auc"])
                    if np.isfinite(new_row.get("incremental_information_delta_auc", np.nan))
                    else float("nan")
                )
                raw_val = _metric_or_nan(new_row["delta_r_raw"])
                new_row["delta_r"] = float(raw_val)

                return new_row

            # Evaluate Tier-1
            surviving_tier1 = []
            for c_info in tier1_candidates:
                eval_res = eval_candidate(c_info, tier=1)
                if eval_res is not None:
                    surviving_tier1.append((c_info, eval_res))

            # ---------------------------------------------------------
            # 3B. Select Top Single Regimes
            # ---------------------------------------------------------
            surviving_tier1.sort(key=lambda x: x[1].get("net_regime_value", 0.0), reverse=True)
            top_tier1 = surviving_tier1[:max_singles]

            for c_info, eval_res in top_tier1:
                cond_rows.append(eval_res)

            # ---------------------------------------------------------
            # 3C. Generate Two-Regime Candidates (Tier-2)
            # ---------------------------------------------------------
            tier2_candidates = []

            for i in range(len(top_tier1)):
                for j in range(i + 1, len(top_tier1)):
                    if len(tier2_candidates) >= max_pairs:
                        break
                    c1_info, r1 = top_tier1[i]
                    c2_info, r2 = top_tier1[j]

                    # Avoid redundant pairs (same feature)
                    if set(c1_info["features"]).intersection(set(c2_info["features"])):
                        continue

                    # Prefer cross-family combinations (skip if same family)
                    if set(c1_info["families"]).intersection(set(c2_info["families"])):
                        continue

                    combined_mask = c1_info["mask"] & c2_info["mask"]
                    # Determine best parent for relative comparison
                    best_parent_res = r1 if r1.get("net_regime_value", 0) > r2.get("net_regime_value", 0) else r2

                    tier2_candidates.append({
                        "name": f"{c1_info['name']}_AND_{c2_info['name'].replace(cand_name + '_', '')}",
                        "desc": f"{c1_info['desc']} AND {c2_info['desc']}",
                        "mask": combined_mask,
                        "features": c1_info["features"] + c2_info["features"],
                        "families": c1_info["families"] + c2_info["families"],
                        "best_parent_res": best_parent_res
                    })

            for c_info in tier2_candidates:
                parent_res = c_info.pop("best_parent_res")
                eval_res = eval_candidate(c_info, tier=2, parent_res=parent_res)
                if eval_res is not None:
                    cond_rows.append(eval_res)

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
        cond_feature_gain: List[np.float32] = []
        cond_feature_pos: List[np.float32] = []
        cond_top_lifts: List[str] = []
        cond_pred_gain: List[np.float32] = []
        cond_pred_pos: List[np.float32] = []
        cond_pred_regime_r2: List[np.float32] = []
        cond_pred_base_r2: List[np.float32] = []
        cond_spread_vals: List[np.float32] = []
        cond_econ: List[np.float32] = []
        cond_cov: List[np.float32] = []
        cond_fixed_cov: List[np.float32] = []
        cond_geom: List[Any] = []
        cond_tbm_auc_regime: List[np.float32] = []
        cond_tbm_auc_base: List[np.float32] = []
        cond_tbm_auc_lift: List[np.float32] = []
        cond_tbm_top_lift: List[np.float32] = []
        cond_tbm_pos: List[np.float32] = []
        cond_tbm_stability: List[np.float32] = []
        cond_tbm_geom_name: List[str] = []
        for _, row in df_short.iterrows():
            masks = candidate_masks[str(row["name"])]
            side_mask = _get_side_mask(mode, masks["m_high"], masks["m_low"])
            feat_metrics = _compute_phase3_feature_learnability(
                shared, feature_dict, side_mask, mode, folds, cfg
            )
            cond_metrics = _compute_conditional_predictability_metrics(
                shared, side_mask, mode, folds, cfg
            )
            econ_metrics = _compute_tbm_economic_gain(
                shared, side_mask, mode, folds, cfg
            )
            mfe_metrics = _compute_mfe_coverage(shared, side_mask, cfg)
            tbm_lgbm_metrics = _compute_phase4_tbm_lgbm_metrics(
                shared,
                side_mask,
                folds,
                cfg,
                econ_metrics["per_geometry_metrics"],
            )
            cond_feature_gain.append(
                np.float32(feat_metrics["feature_learnability_gain"])
            )
            cond_feature_pos.append(
                np.float32(feat_metrics["feature_positive_fold_fraction"])
            )
            cond_top_lifts.append(str(feat_metrics["top_feature_lifts"]))
            cond_pred_gain.append(
                np.float32(cond_metrics["conditional_predictability_gain"])
            )
            cond_pred_pos.append(
                np.float32(
                    cond_metrics["conditional_predictability_positive_fold_fraction"]
                )
            )
            cond_pred_regime_r2.append(
                np.float32(cond_metrics["conditional_predictability_regime_r2"])
            )
            cond_pred_base_r2.append(
                np.float32(cond_metrics["conditional_predictability_baseline_r2"])
            )
            cond_spread_vals.append(np.float32(cond_metrics["feature_conditioned_spread"]))
            cond_econ.append(np.float32(econ_metrics["economic_gain_r"]))
            cond_cov.append(np.float32(econ_metrics["geometry_weighted_mfe_coverage"]))
            cond_fixed_cov.append(np.float32(mfe_metrics["fixed_tp_mfe_coverage"]))
            cond_geom.append(econ_metrics["per_geometry_metrics"])
            cond_tbm_auc_regime.append(
                np.float32(tbm_lgbm_metrics["tbm_lgbm_auc_regime"])
            )
            cond_tbm_auc_base.append(
                np.float32(tbm_lgbm_metrics["tbm_lgbm_auc_baseline"])
            )
            cond_tbm_auc_lift.append(
                np.float32(tbm_lgbm_metrics["tbm_lgbm_auc_lift_vs_baseline"])
            )
            cond_tbm_top_lift.append(
                np.float32(
                    tbm_lgbm_metrics["tbm_lgbm_top_bucket_lift_vs_baseline"]
                )
            )
            cond_tbm_pos.append(
                np.float32(tbm_lgbm_metrics["tbm_lgbm_positive_fold_fraction"])
            )
            cond_tbm_stability.append(
                np.float32(tbm_lgbm_metrics["tbm_lgbm_stability"])
            )
            cond_tbm_geom_name.append(
                str(tbm_lgbm_metrics["tbm_lgbm_selected_geometry"])
            )

        df_short["feature_learnability_gain"] = np.asarray(
            cond_feature_gain, dtype=np.float32
        )
        df_short["feature_positive_fold_fraction"] = np.asarray(
            cond_feature_pos, dtype=np.float32
        )
        df_short["top_feature_lifts"] = cond_top_lifts
        df_short["conditional_predictability_gain"] = np.asarray(
            cond_pred_gain, dtype=np.float32
        )
        df_short["conditional_predictability_positive_fold_fraction"] = np.asarray(
            cond_pred_pos, dtype=np.float32
        )
        df_short["conditional_predictability_regime_r2"] = np.asarray(
            cond_pred_regime_r2, dtype=np.float32
        )
        df_short["conditional_predictability_baseline_r2"] = np.asarray(
            cond_pred_base_r2, dtype=np.float32
        )
        df_short["feature_conditioned_spread"] = np.asarray(
            cond_spread_vals, dtype=np.float32
        )
        df_short["economic_gain_r"] = np.asarray(cond_econ, dtype=np.float32)
        df_short["geometry_weighted_mfe_coverage"] = np.asarray(
            cond_cov, dtype=np.float32
        )
        df_short["fixed_tp_mfe_coverage"] = np.asarray(cond_fixed_cov, dtype=np.float32)
        df_short["aggregate_mfe_coverage"] = df_short[
            "geometry_weighted_mfe_coverage"
        ].astype(np.float32)
        df_short["per_geometry_metrics"] = cond_geom
        df_short["tbm_lgbm_auc_regime"] = np.asarray(
            cond_tbm_auc_regime, dtype=np.float32
        )
        df_short["tbm_lgbm_auc_baseline"] = np.asarray(
            cond_tbm_auc_base, dtype=np.float32
        )
        df_short["tbm_lgbm_auc_lift_vs_baseline"] = np.asarray(
            cond_tbm_auc_lift, dtype=np.float32
        )
        df_short["tbm_lgbm_top_bucket_lift_vs_baseline"] = np.asarray(
            cond_tbm_top_lift, dtype=np.float32
        )
        df_short["tbm_lgbm_positive_fold_fraction"] = np.asarray(
            cond_tbm_pos, dtype=np.float32
        )
        df_short["tbm_lgbm_stability"] = np.asarray(
            cond_tbm_stability, dtype=np.float32
        )
        df_short["tbm_lgbm_selected_geometry"] = cond_tbm_geom_name
        coverage_ref = float(max(cfg.get("mask_opt_min_mfe_coverage", 0.02), 0.25))
        df_short["coverage_multiplier"] = (
            0.25
            + 0.75
            * np.clip(
                df_short["aggregate_mfe_coverage"].astype(np.float32).values
                / max(coverage_ref, 1e-6),
                0.0,
                1.0,
            )
        ).astype(np.float32)
        df_short["learnability_support"] = np.maximum(
            df_short["incremental_information_delta_auc"].astype(np.float32).values,
            0.0,
        ).astype(np.float32)
        df_short["noise_penalty"] = (
            1.0
            + np.log1p(
                np.maximum(
                    np.nan_to_num(
                        df_short["dispersion_to_edge_ratio"].astype(np.float32).values,
                        nan=100.0,
                    ),
                    0.0,
                )
            )
        ).astype(np.float32)
        df_short["effective_edge"] = (
            np.maximum(df_short["delta_r_shrunk"].astype(np.float32).values, 0.0)
            * np.maximum(df_short["S_r"].astype(np.float32).values, 0.0)
            * df_short["primary_multiplier"].astype(np.float32).values
            * df_short["worst_fold_multiplier"].astype(np.float32).values
            * df_short["disagreement_penalty"].astype(np.float32).values
        ).astype(np.float32)
        df_short["score_r"] = (
            df_short["effective_edge"].astype(np.float32).values
            * (1.0 + 25.0 * df_short["learnability_support"].astype(np.float32).values)
            / np.maximum(df_short["noise_penalty"].astype(np.float32).values, 1e-6)
        ).astype(np.float32)
        df_short["predictability_anchor"] = np.maximum(
            df_short["conditional_predictability_gain"].astype(np.float32).values, 0.0
        ).astype(np.float32)
        df_short["predictability_positive_multiplier"] = (
            0.75
            + 0.25
            * df_short[
                "conditional_predictability_positive_fold_fraction"
            ].astype(np.float32).values
        ).astype(np.float32)
        df_short["spread_multiplier"] = (
            1.0
            + np.tanh(10.0 * df_short["feature_conditioned_spread"].astype(np.float32).values)
        ).astype(np.float32)
        df_short["difference_prior"] = (
            0.85 + 0.15 * np.clip(df_short["score_r"].astype(np.float32).values, 0.0, None)
        ).astype(np.float32)
        df_short["score_ml"] = (
            df_short["score_r"].astype(np.float32).values
            * (
                1.0
                + 5.0 * df_short["predictability_anchor"].astype(np.float32).values
            )
            * df_short["predictability_positive_multiplier"].astype(np.float32).values
            * df_short["spread_multiplier"].astype(np.float32).values
            * df_short["difference_prior"].astype(np.float32).values
        ).astype(np.float32)
        df_short["tbm_auc_support"] = np.maximum(
            df_short["tbm_lgbm_auc_lift_vs_baseline"].astype(np.float32).values, 0.0
        ).astype(np.float32)
        df_short["tbm_lift_support"] = np.maximum(
            df_short["tbm_lgbm_top_bucket_lift_vs_baseline"].astype(np.float32).values,
            0.0,
        ).astype(np.float32)
        df_short["score_ml_trading"] = (
            df_short["score_ml"].astype(np.float32).values
            * (
                1.0
                + 2.0 * np.maximum(df_short["feature_learnability_gain"].astype(np.float32).values, 0.0)
            )
            * (1.0 + 25.0 * df_short["tbm_auc_support"].astype(np.float32).values)
            * (1.0 + 10.0 * df_short["tbm_lift_support"].astype(np.float32).values)
            * (
                0.50
                + 0.50
                * np.maximum(
                    df_short["tbm_lgbm_stability"].astype(np.float32).values, 0.0
                )
            )
            * (
                0.50
                + 0.50
                * np.maximum(
                    df_short["tbm_lgbm_positive_fold_fraction"].astype(np.float32).values,
                    0.0,
                )
            )
            * df_short["coverage_multiplier"].astype(np.float32).values
        ).astype(np.float32)
        df_short["shortlist_score"] = df_short["score_ml_trading"].astype(np.float32)
        # Apply complexity penalties
        phase4_single_regime_penalty = float(cfg.get("phase4_single_regime_penalty", 0.95))
        phase4_two_regime_penalty = float(cfg.get("phase4_two_regime_penalty", 0.85))

        penalties = []
        for _, row in df_short.iterrows():
            tier = row.get("tier", 0)
            if tier == 1:
                penalties.append(phase4_single_regime_penalty)
            elif tier == 2:
                penalties.append(phase4_two_regime_penalty)
            else:
                penalties.append(1.0)

        df_short["complexity_multiplier"] = np.array(penalties, dtype=np.float32)
        df_short["score_ml_trading"] = df_short["score_ml_trading"].astype(np.float32).values * df_short["complexity_multiplier"].astype(np.float32).values
        df_short["shortlist_score"] = df_short["score_ml_trading"].astype(np.float32)

        # Dominance Pruning
        base_rows = df_short[df_short["tier"] == 0].copy()

        keep_idx: List[int] = []
        tolerance = float(cfg.get("phase4_dominance_tolerance", 0.90))

        # We process each candidate and see if a simpler candidate strictly dominates it
        for idx, row in df_short.iterrows():
            tier = row.get("tier", 0)
            if tier == 0:
                keep_idx.append(idx)
                continue

            base_name = str(row["name"]).split("_" + row["conditioner_mode"].replace(" ", "").replace(">", "gt").replace("<", "lt"))[0]
            if "AND" in str(row["name"]):
                base_name = str(row["name"]).split("_AND_")[0].rsplit("_", 1)[0]

            dominated = False
            # Compare against all simpler candidates of the same base
            simpler_cands = df_short[(df_short["tier"] < tier)]

            for _, s_row in simpler_cands.iterrows():
                # Rough check if they share the same base name (ignoring conditioner suffixes)
                if not str(s_row["name"]).startswith(base_name):
                    continue

                # A dominates B if:
                if (
                    _metric_or_nan(s_row.get("score_ml_trading")) >= _metric_or_nan(row.get("score_ml_trading")) and
                    _metric_or_nan(s_row.get("economic_gain_r")) >= _metric_or_nan(row.get("economic_gain_r")) and
                    _metric_or_nan(s_row.get("S_r")) >= _metric_or_nan(row.get("S_r")) and
                    _metric_or_nan(s_row.get("total_events")) >= _metric_or_nan(row.get("total_events")) * tolerance
                ):
                    dominated = True
                    break

            if not dominated:
                keep_idx.append(idx)

        df_short = (
            df_short.loc[keep_idx]
            .sort_values("score_ml_trading", ascending=False)
            .copy()
        )

    _log_stage_snapshot(
        mode,
        "Phase 4",
        df_short,
        "score_ml_trading",
        [
            "name",
            "score_ml_trading",
            "tbm_lgbm_auc_lift_vs_baseline",
            "tbm_lgbm_stability",
            "tbm_lgbm_positive_fold_fraction",
            "tbm_lgbm_top_bucket_lift_vs_baseline",
            "tbm_lgbm_selected_geometry",
            "aggregate_mfe_coverage",
            "score_ml",
            "delta_r_raw",
        ],
    )

    final_diag_k = int(cfg.get("final_top_k_for_diagnostics", 3))

    # Jaccard diversity selection
    df_short = df_short.sort_values("shortlist_score", ascending=False).reset_index(drop=True)
    selected_idx = []
    selected_masks = []

    for idx, row in df_short.iterrows():
        if len(selected_idx) >= final_diag_k:
            break

        m_info = candidate_masks.get(str(row["name"]), {})
        if not m_info:
            continue

        side_mask = _get_side_mask(mode, m_info.get("m_high", np.array([])), m_info.get("m_low", np.array([])))

        # Check Jaccard similarity against already selected
        is_diverse = True
        for sel_mask in selected_masks:
            intersection = np.sum(side_mask & sel_mask)
            union = np.sum(side_mask | sel_mask)
            jaccard = intersection / max(union, 1)
            if jaccard > 0.80:  # Too similar
                is_diverse = False
                break

        if is_diverse:
            selected_idx.append(idx)
            selected_masks.append(side_mask)

    if not selected_idx and not df_short.empty:
        selected_idx = [0]

    df_diag_input = df_short.loc[selected_idx].copy()
    df_diag = _final_topk_diagnostics(
        mode, df_diag_input, candidate_masks, shared, feature_dict, cfg
    )

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
        return {
            "status": "failed",
            "reason": f"mode_crashed_{mode}_exit_{proc.exitcode}",
        }
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
    return optimize_layer_masks_by_mode(
        data, feature_dict, forward_returns, cfg, modes=modes, layer_name="layer0"
    )


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
    isolate_modes = (
        bool(runtime_cfg.get("mask_opt_isolate_modes", True)) and len(modes) > 1
    )
    shared: Optional[Dict[str, Any]] = None
    if not isolate_modes:
        shared = _build_shared_cache(data, feature_dict, forward_returns, runtime_cfg)

    for mode in modes:
        if isolate_modes:
            tprint(f"Running mode {mode} in isolated subprocess...")
            res = _run_mode_search_isolated(
                mode, data, feature_dict, forward_returns, runtime_cfg
            )
        else:
            assert shared is not None
            res = _run_mode_search(mode, shared, feature_dict, runtime_cfg)
        mode_results[mode] = res

        if res.get("status") == "ok":
            best = res["layer0_best_config_"]
            primary_col = _mode_primary_predictability_col(mode)
            summary_rows.append(
                {
                    "mode": mode,
                    "status": "ok",
                    "best_name": str(best.get("name", "")),
                    "candidate_count": len(res["layer0_candidate_table_"]),
                    "shortlist_count": len(res["layer0_shortlist_"]),
                    "best_shortlist_score": float(best.get("shortlist_score", 0.0)),
                    "score_r": float(
                        best.get("score_r", best.get("shortlist_score", 0.0))
                    ),
                    "delta_r": _metric_or_nan(best.get("delta_r")),
                    "N_r": float(best.get("N_r", best.get("total_events", 0.0))),
                    "S_r": _metric_or_nan(best.get("S_r")),
                    "D_r": _metric_or_nan(best.get("D_r")),
                    "event_count": int(best.get("total_events", 0)),
                    "active_days_fraction": float(
                        best.get("active_days_fraction", 0.0)
                    ),
                    "events_per_day_mean": _metric_or_nan(
                        best.get("events_per_day_mean")
                    ),
                    "events_per_day_std": _metric_or_nan(
                        best.get("events_per_day_std")
                    ),
                    "events_per_day_per_asset": _metric_or_nan(
                        best.get("events_per_day_per_asset")
                    ),
                    "primary_gain": _metric_or_nan(best.get(primary_col)),
                    "primary_gain_is_nan": _metric_or_nan(
                        best.get("primary_predictability_gain_is_nan")
                    ),
                    "incremental_information_delta_auc": _metric_or_nan(
                        best.get("incremental_information_delta_auc")
                    ),
                    "incremental_information_positive_fold_fraction": _metric_or_nan(
                        best.get("incremental_information_positive_fold_fraction")
                    ),
                    "dispersion_to_edge_ratio": _metric_or_nan(
                        best.get("dispersion_to_edge_ratio")
                    ),
                    "selected_delta_metric": str(best.get("selected_delta_metric", "")),
                    "decision": str(best.get("decision", "ranked")),
                    "rationale": str(best.get("rationale", "")),
                }
            )
        else:
            summary_rows.append(
                {
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
                }
            )

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
    cfg["mask_opt_target_event_density"] = 0.012
    cfg["mask_opt_min_events_floor"] = 150
    cfg["mask_opt_min_active_days_floor"] = 0.25
    cfg["mask_opt_min_mfe_coverage"] = float(cfg.get("mask_opt_min_mfe_coverage", 0.02))
    cfg["mask_opt_min_primary_gain"] = float(cfg.get("mask_opt_min_primary_gain", 0.005))
    cfg["mask_opt_max_dispersion_to_edge_ratio"] = float(
        cfg.get("mask_opt_max_dispersion_to_edge_ratio", 20.0)
    )
    cfg["min_positive_fold_fraction"] = 0.60
    cfg["shortlist_max_candidates"] = 4
    cfg["final_top_k_for_diagnostics"] = 3
    cfg["mask_opt_max_rows"] = 50_000        # Phase 2: full metric evaluation
    cfg["mask_opt_deep_rows"] = 150_000      # Phases 3+4: TBM econ + diagnostics
    cfg["mask_opt_pre_slice_max_rows"] = int(
        cfg.get("mask_opt_pre_slice_max_rows", 1_000_000)
    )
    cfg["phase1_classifier_max_samples_per_class"] = int(
        cfg.get("phase1_classifier_max_samples_per_class", 15_000)
    )
    cfg["phase2_metric_max_samples_per_class"] = int(
        cfg.get("phase2_metric_max_samples_per_class", 25_000)
    )

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

    store = PartitionedOHLCVStore(
        root_dir=cfg["data_root"], timeframe=cfg.get("timeframe", "1h")
    )

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

    symbols = list(all_symbols)
    if args.max_symbols is not None:
        symbols = symbols[: max(1, int(args.max_symbols))]
        tprint(f"Selected {len(symbols)} symbols via --max-symbols")
    else:
        tprint(f"Selected {len(symbols)} deduplicated symbols")

    start_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(
        days=int(365.25 * args.lookback_years)
    )

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
    fwd_ret_wide = (
        panel["close"].pct_change(fwd_hours, fill_method=None).shift(-fwd_hours)
    )

    n_timestamps = len(common_idx)
    n_symbols = len(common_syms)
    data_stacked = pd.DataFrame(
        {
            "timestamp": np.repeat(common_idx.to_numpy(), n_symbols),
            "symbol": np.tile(common_syms.to_numpy(dtype=object), n_timestamps),
            "close": _flatten_wide_frame(panel["close"], common_idx, common_syms),
            "high": _flatten_wide_frame(panel["high"], common_idx, common_syms),
            "low": _flatten_wide_frame(panel["low"], common_idx, common_syms),
        }
    )

    feature_dict: Dict[str, np.ndarray] = {}
    for k, df in feat_dict_raw.items():
        if isinstance(df, pd.DataFrame):
            arr = _flatten_wide_frame(df, common_idx, common_syms)
            arr[np.isinf(arr)] = np.nan
            feature_dict[k] = arr.astype(np.float32)

    fwd_ret_stacked = _flatten_wide_frame(fwd_ret_wide, common_idx, common_syms)

    tprint(f"Total rows available before SlicePlanner: {data_stacked.shape[0]}")
    pre_slice_max_rows = int(cfg.get("mask_opt_pre_slice_max_rows", 1_000_000))
    if data_stacked.shape[0] > pre_slice_max_rows:
        start_idx = data_stacked.shape[0] - pre_slice_max_rows
        data_stacked = data_stacked.iloc[start_idx:].reset_index(drop=True)
        fwd_ret_stacked = fwd_ret_stacked[start_idx:]
        for k in feature_dict:
            feature_dict[k] = feature_dict[k][start_idx:]
        tprint(
            f"Capped pre-SlicePlanner rows to {data_stacked.shape[0]} for mask optimization."
        )
    # Apply regime_search slice plan on the full dataset FIRST.
    # This gives temporally-structured rows spanning full history (~150K rows).
    # Phase 1/2 then cap this structured sample to 50K / 20K respectively.
    # Phase 3+4 use the full SlicePlanner result (up to mask_opt_deep_rows=150K).
    deep_data, deep_feature_dict, deep_fwd_ret = _apply_regime_search_slice_plan(
        data=data_stacked,
        feature_dict=feature_dict,
        forward_returns=fwd_ret_stacked,
        lookback_years=float(args.lookback_years),
    )
    deep_rows = int(cfg.get("mask_opt_deep_rows", 150_000))
    if deep_data.shape[0] > deep_rows:
        deep_data = deep_data.iloc[-deep_rows:].reset_index(drop=True)
        deep_fwd_ret = deep_fwd_ret[-deep_rows:]
        for k in deep_feature_dict:
            deep_feature_dict[k] = deep_feature_dict[k][-deep_rows:]
    tprint(f"SlicePlanner gave {deep_data.shape[0]} structured rows for Phase 3+4.")

    if args.mode == "all":
        modes = ALL_MODES[:]
    else:
        modes = [args.mode]

    tprint(f"Starting 4-mode Layer 0 optimization (deep={deep_data.shape[0]}, cap={cfg.get('mask_opt_max_rows', 50_000)}, p1_floor={cfg.get('phase1_min_subsample_rows', 20_000)})...")
    result = optimize_layer0_masks_by_mode(
        deep_data, deep_feature_dict, deep_fwd_ret, cfg, modes=modes
    )

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
            INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV,
            REPORTS_DIR,
            save_best_params_csv,
        )

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = REPORTS_DIR / "inference_candidate_mask_mode_summary.csv"
        result["mode_summary_table_"].to_csv(summary_path, index=False)
        bucket_winner_rows: List[Dict[str, Any]] = []
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
                bucket_row = dict(best)
                bucket_row["bucket"] = mode
                bucket_row["mode"] = mode
                bucket_winner_rows.append(bucket_row)
        if bucket_winner_rows:
            bucket_winners_path = (
                REPORTS_DIR / "inference_candidate_mask_best_params_per_bucket.csv"
            )
            pd.DataFrame(bucket_winner_rows).to_csv(bucket_winners_path, index=False)
    except Exception as e:
        tprint(f"Warning: failed to save best params: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize Layer 0 masks in 4 explicit modes"
    )
    parser.add_argument("--data-root", help="Override data root")
    parser.add_argument("--features", help="Path to features directory")
    parser.add_argument("--perps", action="store_true", help="Use perpetual mode data")
    parser.add_argument("--max-symbols", type=int, help="Cap symbols for speed")
    parser.add_argument(
        "--lookback-years", type=float, default=2.0, help="Years of data to load"
    )
    parser.add_argument(
        "--mode",
        choices=["all"] + ALL_MODES,
        default="all",
        help="Run all modes or one mode only",
    )
    args = parser.parse_args()
    run_mask_optimization_4modes(args)
