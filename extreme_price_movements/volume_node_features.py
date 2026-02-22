"""
HVN/LVN-focused, single-asset OHLCV, fully vectorised (no row loops).
Computes a rolling volume-by-price profile in a stationary z-price space, then derives HVN/LVN features.

Key idea (vectorised + stable across price scales):
1) Define z_t = (typical_price - rolling_VWAP) / rolling_std(typical_price) over vp_lookback.
2) For each rolling window, build volume histogram over fixed z-bins.
3) HVN(s) = high-volume bins; LVN(s) = low-volume (nonzero) bins.
4) Map selected z-centers back to price: price = vwap + z_center * std.
5) Produce distances in ATR units + zone/touch/acceptance metrics.

Inputs:
- df columns: open, high, low, close, volume
- constant bar interval assumed
- heavy but loop-free; sliding_window_view can be memory intensive.

Max features produced here: 18 (all HVN/LVN-centric).
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

def hvn_lvn_features_ohlcv(
    df: pd.DataFrame,
    vp_lookback: int = 168,     # e.g., 7d of 1h bars
    vp_bins: int = 21,          # histogram bins in z-space
    topk_hvn: int = 5,          # candidate HVNs to pick nearest above/below
    botk_lvn: int = 8,          # candidate LVNs to pick nearest above/below
    w_atr: int = 14,
    zone_atr: float = 0.5,
    touch_win: int = 12,        # acceptance window (bars)
) -> pd.DataFrame:
    eps = 1e-12
    # Ensure inputs are float
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)

    tp = (h + l + c) / 3.0

    # --- ATR (scale distances) ---
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(w_atr).mean()

    # --- Rolling VWAP & std (for stationary z-space) ---
    v_sum = v.rolling(vp_lookback).sum()
    vwap = (tp.mul(v).rolling(vp_lookback).sum()) / (v_sum + eps)
    tp_std = tp.rolling(vp_lookback).std()

    z = (tp - vwap) / (tp_std + eps)

    # Fixed z-bin edges/centers
    zmin, zmax = -3.0, 3.0
    edges = np.linspace(zmin, zmax, vp_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0  # (vp_bins,)

    # Prepare arrays
    z_np = z.to_numpy()
    v_np = v.to_numpy()
    tp_std_np = tp_std.to_numpy()
    vwap_np = vwap.to_numpy()
    atr_np = atr.to_numpy()
    c_np = c.to_numpy()

    n = len(df)
    m = n - vp_lookback + 1

    # Outputs aligned to window end index
    out_end = np.arange(vp_lookback - 1, n)

    # Placeholders (in z-space)
    poc_z = np.full(n, np.nan)            # primary HVN (POC)
    hvn_above_z = np.full(n, np.nan)
    hvn_below_z = np.full(n, np.nan)
    lvn_above_z = np.full(n, np.nan)
    lvn_below_z = np.full(n, np.nan)

    # Profile stats
    bin_share = np.full(n, np.nan)        # volume share at current bin
    prof_conc = np.full(n, np.nan)        # max(hist)/sum(hist)
    prof_entropy = np.full(n, np.nan)     # entropy(hist/sum)
    lvn_depth = np.full(n, np.nan)        # min_nonzero(hist)/mean(hist_nonzero) (low means "thin")

    if n >= vp_lookback:
        zw = sliding_window_view(z_np, vp_lookback)     # (m, vp_lookback)
        vw = sliding_window_view(v_np, vp_lookback)     # (m, vp_lookback)

        # Bin indices in [0, vp_bins-1]
        idx = np.digitize(zw, edges) - 1
        idx = np.clip(idx, 0, vp_bins - 1)

        # Prepare histogram: (m, vp_bins)
        # Avoid huge expansion (m, vp_lookback, vp_bins)
        hist = np.zeros((m, vp_bins))
        
        # We want to add vw_i to hist[i, idx_i] for all i and window_position
        # idx has shape (m, vp_lookback)
        # vw has shape (m, vp_lookback)
        rows = np.arange(m)[:, None] # (m, 1)
        # idx[rows, cols] maps to hist[rows, idx]
        # We can use np.add.at for unbuffered in-place addition, but it's slow.
        # Vectorized alternative:
        flat_idx = idx.ravel()
        flat_rows = np.repeat(np.arange(m), vp_lookback)
        flat_vol = vw.ravel()
        
        # Using bincount with weights for high speed if we can map (row, bin) to a flat index
        # global_bin_idx = row * vp_bins + bin
        combined_idx = flat_rows * vp_bins + flat_idx
        hist_flat = np.bincount(combined_idx, weights=flat_vol, minlength=m * vp_bins)
        hist = hist_flat.reshape(m, vp_bins)

        hist_sum = hist.sum(axis=1) + eps
        p = hist / hist_sum[:, None]

        # Primary HVN (POC)
        poc_bin = np.argmax(hist, axis=1)
        poc_z[out_end] = centers[poc_bin]

        # Current z at window end, and its bin
        z0 = z_np[out_end]
        z0_bin = np.clip(np.digitize(z0, edges) - 1, 0, vp_bins - 1)

        # Volume share at current bin
        # indexing with (arange(m), z0_bin)
        bin_vol = hist[np.arange(m), z0_bin]
        bin_share[out_end] = bin_vol / hist_sum

        # Concentration + entropy
        prof_conc[out_end] = hist.max(axis=1) / hist_sum
        # entropy with safe log
        prof_entropy[out_end] = -(p * np.log(p + eps)).sum(axis=1)

        # LVN "depth" (thinness)
        hist_nz = hist.copy()
        hist_nz[hist_nz <= 0] = np.nan
        min_nz = np.nanmin(hist_nz, axis=1)
        mean_nz = np.nanmean(hist_nz, axis=1) + eps
        lvn_depth[out_end] = min_nz / mean_nz

        # --- Nearest HVN above/below from top-k bins ---
        k = min(topk_hvn, vp_bins)
        # argpartition ensures the k-th element is in position k, and smaller elements before it
        # We want largest volume, so -hist
        top_idx = np.argpartition(-hist, kth=k-1, axis=1)[:, :k]  # (m, k)
        top_cent = centers[top_idx]                                # (m, k)

        dz_top = top_cent - z0[:, None]
        # above: smallest positive dz
        above_mask = dz_top > 0
        dz_above = np.where(above_mask, dz_top, np.inf)
        pick_above = np.argmin(dz_above, axis=1)
        hvn_above = top_cent[np.arange(m), pick_above]

        # Check if we actually found one
        valid_above = ~np.isinf(dz_above[np.arange(m), pick_above])
        hvn_above[~valid_above] = np.nan

        # below: largest negative dz (closest below)
        # we want max(dz) where dz < 0. or min(|dz|) where dz < 0.
        below_mask = dz_top < 0
        dz_below = np.where(below_mask, dz_top, -np.inf)
        pick_below = np.argmax(dz_below, axis=1)
        hvn_below = top_cent[np.arange(m), pick_below]

        valid_below = ~np.isneginf(dz_below[np.arange(m), pick_below])
        hvn_below[~valid_below] = np.nan

        hvn_above_z[out_end] = hvn_above
        hvn_below_z[out_end] = hvn_below

        # --- Nearest LVN above/below from bottom-k (nonzero) bins ---
        hist_lvn = hist.copy()
        hist_lvn[hist_lvn <= 0] = np.inf # ignore zero bins for LVN? or treat them as best LVN?
        # "LVN(s) = low-volume (nonzero) bins" -> User spec says nonzero.

        kb = min(botk_lvn, vp_bins)
        bot_idx = np.argpartition(hist_lvn, kth=kb-1, axis=1)[:, :kb]  # (m, kb)
        bot_cent = centers[bot_idx]                                     # (m, kb)

        dz_bot = bot_cent - z0[:, None]
        above_mask = dz_bot > 0
        dz_above = np.where(above_mask, dz_bot, np.inf)
        pick_above = np.argmin(dz_above, axis=1)
        lvn_above = bot_cent[np.arange(m), pick_above]

        valid_above = ~np.isinf(dz_above[np.arange(m), pick_above])
        lvn_above[~valid_above] = np.nan

        below_mask = dz_bot < 0
        dz_below = np.where(below_mask, dz_bot, -np.inf)
        pick_below = np.argmax(dz_below, axis=1)
        lvn_below = bot_cent[np.arange(m), pick_below]

        valid_below = ~np.isneginf(dz_below[np.arange(m), pick_below])
        lvn_below[~valid_below] = np.nan

        lvn_above_z[out_end] = lvn_above
        lvn_below_z[out_end] = lvn_below

    # --- Map z-levels -> price levels: price = vwap + z * std ---
    poc_px = vwap_np + poc_z * tp_std_np
    hvn_above_px = vwap_np + hvn_above_z * tp_std_np
    hvn_below_px = vwap_np + hvn_below_z * tp_std_np
    lvn_above_px = vwap_np + lvn_above_z * tp_std_np
    lvn_below_px = vwap_np + lvn_below_z * tp_std_np

    atr_safe = atr_np + eps

    # --- Distances (ATR units; signed: + if price above level) ---
    # User spec: "produce distances in ATR units".
    # Typically dist = (Price - Level) / ATR.
    # Note: "signed: + if price above level" -> (c - level).

    dist_poc = (c_np - poc_px) / atr_safe
    dist_hvn_above = (c_np - hvn_above_px) / atr_safe
    dist_hvn_below = (c_np - hvn_below_px) / atr_safe
    dist_lvn_above = (c_np - lvn_above_px) / atr_safe
    dist_lvn_below = (c_np - lvn_below_px) / atr_safe

    # --- Zones (within zone_atr ATR of the level) ---
    in_poc = (np.abs(c_np - poc_px) / atr_safe < zone_atr).astype(float)
    in_hvn_above = (np.abs(c_np - hvn_above_px) / atr_safe < zone_atr).astype(float)
    in_hvn_below = (np.abs(c_np - hvn_below_px) / atr_safe < zone_atr).astype(float)
    in_lvn_above = (np.abs(c_np - lvn_above_px) / atr_safe < zone_atr).astype(float)
    in_lvn_below = (np.abs(c_np - lvn_below_px) / atr_safe < zone_atr).astype(float)

    # --- "Acceptance" proxies near nodes (touch frequency over recent window) ---
    # (Still HVN/LVN-centric, no L2 needed)
    accept_poc = pd.Series(in_poc, index=df.index).rolling(touch_win).mean().to_numpy()
    accept_hvn = pd.Series(np.nan_to_num(in_hvn_above + in_hvn_below), index=df.index).rolling(touch_win).mean().to_numpy()
    accept_lvn = pd.Series(np.nan_to_num(in_lvn_above + in_lvn_below), index=df.index).rolling(touch_win).mean().to_numpy()

    # --- Assemble (<= 20 features; here: 18) ---
    feats = pd.DataFrame(index=df.index)
    feats["dist_poc_atr"] = dist_poc
    feats["dist_hvn_above_atr"] = dist_hvn_above
    feats["dist_hvn_below_atr"] = dist_hvn_below
    feats["dist_lvn_above_atr"] = dist_lvn_above
    feats["dist_lvn_below_atr"] = dist_lvn_below

    feats["in_poc_zone"] = in_poc
    feats["in_hvn_above_zone"] = in_hvn_above
    feats["in_hvn_below_zone"] = in_hvn_below
    feats["in_lvn_above_zone"] = in_lvn_above
    feats["in_lvn_below_zone"] = in_lvn_below

    feats["bin_vol_share"] = bin_share
    feats["profile_concentration"] = prof_conc
    feats["profile_entropy"] = prof_entropy
    feats["lvn_depth_ratio"] = lvn_depth  # smaller => thinner pockets

    feats["accept_poc_touchrate"] = accept_poc
    feats["accept_hvn_touchrate"] = accept_hvn
    feats["accept_lvn_touchrate"] = accept_lvn

    # Optional: add "air-pocket vs friction" single scalar (still HVN/LVN-focused)
    # air_pocket_score high when current bin has low share AND profile is not concentrated
    feats["air_pocket_score"] = (1.0 - feats["bin_vol_share"]) * (1.0 - feats["profile_concentration"])

    # Ensure float32
    return feats.astype(np.float32)
