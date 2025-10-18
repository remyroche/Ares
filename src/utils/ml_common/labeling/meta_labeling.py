"""
Lopez de Prado meta-labeling utilities (triple-barrier method, event labeling, purged CV helpers).

This module implements a pragmatic subset of the Advances in Financial Machine Learning (AFML)
triple-barrier labeling suitable for intraday OHLCV data with bar-based horizons.

Design choices:
- Bar-based horizon (horizon_bars) rather than calendar time
- Optional volatility scaling via rolling std of returns
- Supports long-only, short-only, or sided signals
- Returns labels in {0, 1} for meta-labeling (success/failure of primary signal)

Note: This module is intentionally lightweight and avoids external dependencies.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd


def compute_return_series(close: pd.Series) -> pd.Series:
    """Compute log or arithmetic returns (arithmetic here to match other code paths)."""
    return close.pct_change()


def compute_volatility(close: pd.Series, span: int = 50) -> pd.Series:
    """Rolling volatility proxy using returns' EWMA std (span in bars)."""
    ret = compute_return_series(close)
    vol = ret.ewm(span=span, adjust=False).std()
    return vol.fillna(method='bfill').fillna(0)


def _first_barrier_hit(path_vals: np.ndarray, tp_price: float, sl_price: float, is_short: bool) -> Tuple[Optional[int], Optional[int]]:
    """Return indices (relative to path_vals) where TP or SL is first hit; None if not hit."""
    if not is_short:
        up_cross_idx = np.where(path_vals >= tp_price)[0]
        dn_cross_idx = np.where(path_vals <= sl_price)[0]
    else:
        up_cross_idx = np.where(path_vals <= tp_price)[0]  # TP for short is down
        dn_cross_idx = np.where(path_vals >= sl_price)[0]  # SL for short is up

    up_idx = int(up_cross_idx[0]) if up_cross_idx.size > 0 else None
    dn_idx = int(dn_cross_idx[0]) if dn_cross_idx.size > 0 else None
    return up_idx, dn_idx


def triple_barrier_labels(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    horizon_bars: int,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    vol: Optional[pd.Series] = None,
    min_ret: float = 0.0,
    side: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Label events with the triple-barrier method.

    Args:
        close: price series indexed by timestamp
        t_events: event start times (e.g., primary model signals)
        horizon_bars: vertical barrier in bars from event
        pt_mult: profit-take multiplier relative to volatility if vol provided; otherwise absolute fraction
        sl_mult: stop-loss multiplier relative to volatility if vol provided; otherwise absolute fraction
        vol: optional volatility series (same index as close); if None, use fixed absolute fractions
        min_ret: minimum absolute return threshold to keep event (filters tiny moves)
        side: optional Series in {+1, -1} for long/short side per event time; if None, assume +1

    Returns:
        DataFrame indexed by t_events with columns:
            't1'      - vertical barrier time
            'label'   - meta-label in {0,1}
            'ret'     - realized return between event time and vertical barrier
            'pt_hit'  - bool, profit-take barrier was first hit
            'sl_hit'  - bool, stop-loss barrier was first hit
    """
    if not isinstance(close, pd.Series):
        raise ValueError("close must be a Series")
    if not isinstance(t_events, (pd.DatetimeIndex, list)):
        raise ValueError("t_events must be a DatetimeIndex or list of timestamps")

    # Align helpers
    idx_pos = pd.Series(np.arange(len(close)), index=close.index)

    out = {
        't1': [],
        'label': [],
        'ret': [],
        'pt_hit': [],
        'sl_hit': []
    }
    index: List[pd.Timestamp] = []

    for ts in pd.DatetimeIndex(t_events):
        if ts not in close.index:
            continue
        i = int(idx_pos.loc[ts])
        j = min(len(close) - 1, i + horizon_bars)
        if j <= i:
            continue
        s0 = float(close.iloc[i])
        path = close.iloc[i + 1:j + 1].values

        # Determine barrier distances
        if vol is not None and ts in vol.index:
            sigma = float(vol.loc[ts])
            # If vol is near-zero, fallback to min_ret to avoid flat barriers
            px_tp = s0 * (1.0 + max(min_ret, pt_mult * sigma))
            px_sl = s0 * (1.0 - max(min_ret, sl_mult * sigma))
        else:
            px_tp = s0 * (1.0 + max(min_ret, pt_mult))
            px_sl = s0 * (1.0 - max(min_ret, sl_mult))

        d = 1 if side is None or side.get(ts, 1) >= 0 else -1
        up_idx, dn_idx = _first_barrier_hit(path, px_tp, px_sl, is_short=(d < 0))

        pt_first = sl_first = False
        if up_idx is not None and dn_idx is not None:
            pt_first = up_idx < dn_idx
            sl_first = dn_idx < up_idx
        elif up_idx is not None:
            pt_first = True
        elif dn_idx is not None:
            sl_first = True

        # Meta-label: 1 if TP before SL, else 0
        label = 1 if pt_first and not sl_first else 0
        ret = (float(close.iloc[j]) - s0) / s0

        out['t1'].append(close.index[j])
        out['label'].append(int(label))
        out['ret'].append(np.float32(ret))
        out['pt_hit'].append(bool(pt_first))
        out['sl_hit'].append(bool(sl_first))
        index.append(ts)

    df = pd.DataFrame(out, index=pd.DatetimeIndex(index))
    return df


def purged_kfold_splits(
    n_samples: int,
    n_splits: int = 5,
    embargo: int = 0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate purged K-fold splits with optional embargo in bars.

    Returns a list of (train_idx, test_idx) tuples. Purging removes training samples that
    overlap the test fold; embargo removes a buffer of training samples adjacent to test indices.
    """
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    indices = np.arange(n_samples)
    current = 0
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        # Purge + embargo
        left = max(0, start - embargo)
        right = min(n_samples, stop + embargo)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[left:right] = False
        train_idx = indices[train_mask]
        splits.append((train_idx, test_idx))
        current = stop
    return splits

