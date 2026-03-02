import re

with open("extreme_price_movements/optimise_tpsl_ratio.py", "r") as f:
    content = f.read()

# I need to add rC (close return tensor) to EventCache so I can use it in label_from_cache to resolve ambiguities.
# Wait, build_event_cache has rH, rL, rC_end... but not rC for all bars.
# Let's modify EventCache to include rC.

replace_eventcache = """@dataclass
class EventCache:
    event_idx: np.ndarray   # valid event indices (signal time t)
    entry_px: np.ndarray    # (m,)
    rH: np.ndarray          # (m, horizon) high normalized return: H/entry - 1
    rL: np.ndarray          # (m, horizon) low  normalized return: L/entry - 1
    rC: np.ndarray          # (m, horizon) close normalized return: C/entry - 1
    rC_end: np.ndarray      # (m,) close normalized return at horizon end: C_end/entry - 1
    rL_prefix_min: np.ndarray # (m, horizon) normalized low prefix-min for AE (long)
    rH_prefix_max: np.ndarray # (m, horizon) normalized high prefix-max for AE (short)
    horizon: int
    side: str = "long\""""

search_eventcache = """@dataclass
class EventCache:
    event_idx: np.ndarray   # valid event indices (signal time t)
    entry_px: np.ndarray    # (m,)
    rH: np.ndarray          # (m, horizon) high normalized return: H/entry - 1
    rL: np.ndarray          # (m, horizon) low  normalized return: L/entry - 1
    rC_end: np.ndarray      # (m,) close normalized return at horizon end: C_end/entry - 1
    rL_prefix_min: np.ndarray # (m, horizon) normalized low prefix-min for AE (long)
    rH_prefix_max: np.ndarray # (m, horizon) normalized high prefix-max for AE (short)
    horizon: int
    side: str = "long\""""

if search_eventcache in content:
    content = content.replace(search_eventcache, replace_eventcache)
    print("Replaced EventCache")
else:
    print("Could not find EventCache")

# Update build_event_cache
search_build = """    if e.size == 0:
        z = np.zeros((0, HN), dtype=np.float32)
        return EventCache(
            event_idx=np.zeros(0, dtype=np.int32),
            entry_px=np.zeros(0, dtype=np.float32),
            rH=z, rL=z, rC_end=np.zeros(0, dtype=np.float32),
            rL_prefix_min=z,
            rH_prefix_max=z,
            horizon=HN,
            side=side
        )

    # Extract horizon windows
    # shape: (m, HN)
    idx_2d = start[:, None] + np.arange(HN)

    H_win = high[idx_2d]
    L_win = low[idx_2d]
    C_end = close[start + HN - 1]"""

replace_build = """    if e.size == 0:
        z = np.zeros((0, HN), dtype=np.float32)
        return EventCache(
            event_idx=np.zeros(0, dtype=np.int32),
            entry_px=np.zeros(0, dtype=np.float32),
            rH=z, rL=z, rC=z, rC_end=np.zeros(0, dtype=np.float32),
            rL_prefix_min=z,
            rH_prefix_max=z,
            horizon=HN,
            side=side
        )

    # Extract horizon windows
    # shape: (m, HN)
    idx_2d = start[:, None] + np.arange(HN)

    H_win = high[idx_2d]
    L_win = low[idx_2d]
    C_win = close[idx_2d]
    C_end = close[start + HN - 1]"""

if search_build in content:
    content = content.replace(search_build, replace_build)
    print("Replaced build_event_cache part 1")
else:
    print("Could not find build_event_cache part 1")

search_build2 = """    # Returns
    entry_2d = entry_px[:, None]
    rH = (H_win / np.maximum(entry_2d, eps)) - 1.0
    rL = (L_win / np.maximum(entry_2d, eps)) - 1.0
    rC_end = (C_end / np.maximum(entry_px, eps)) - 1.0"""

replace_build2 = """    # Returns
    entry_2d = entry_px[:, None]
    rH = (H_win / np.maximum(entry_2d, eps)) - 1.0
    rL = (L_win / np.maximum(entry_2d, eps)) - 1.0
    rC = (C_win / np.maximum(entry_2d, eps)) - 1.0
    rC_end = (C_end / np.maximum(entry_px, eps)) - 1.0"""

if search_build2 in content:
    content = content.replace(search_build2, replace_build2)
    print("Replaced build_event_cache part 2")
else:
    print("Could not find build_event_cache part 2")

search_build3 = """    return EventCache(
        event_idx=e,
        entry_px=entry_px,
        rH=rH.astype(np.float32, copy=False),
        rL=rL.astype(np.float32, copy=False),
        rC_end=rC_end.astype(np.float32, copy=False),
        rL_prefix_min=rL_prefix_min,
        rH_prefix_max=rH_prefix_max,
        horizon=HN,
        side=side
    )"""

replace_build3 = """    return EventCache(
        event_idx=e,
        entry_px=entry_px,
        rH=rH.astype(np.float32, copy=False),
        rL=rL.astype(np.float32, copy=False),
        rC=rC.astype(np.float32, copy=False),
        rC_end=rC_end.astype(np.float32, copy=False),
        rL_prefix_min=rL_prefix_min,
        rH_prefix_max=rH_prefix_max,
        horizon=HN,
        side=side
    )"""

if search_build3 in content:
    content = content.replace(search_build3, replace_build3)
    print("Replaced build_event_cache part 3")
else:
    print("Could not find build_event_cache part 3")


# Update build_event_cache_15m
search_build_15m = """    if e_1h.size == 0:
        z = np.zeros((0, HN_15m), dtype=np.float32)
        return EventCache(
            event_idx=np.zeros(0, dtype=np.int32),
            entry_px=np.zeros(0, dtype=np.float32),
            rH=z, rL=z, rC_end=np.zeros(0, dtype=np.float32),
            rL_prefix_min=z,
            rH_prefix_max=z,
            horizon=HN_15m,
            side=side
        )

    # Extract windows
    idx_2d = start[:, None] + np.arange(HN_15m)

    H_win = high_15m[idx_2d]
    L_win = low_15m[idx_2d]
    C_end = close_15m[start + HN_15m - 1]"""

replace_build_15m = """    if e_1h.size == 0:
        z = np.zeros((0, HN_15m), dtype=np.float32)
        return EventCache(
            event_idx=np.zeros(0, dtype=np.int32),
            entry_px=np.zeros(0, dtype=np.float32),
            rH=z, rL=z, rC=z, rC_end=np.zeros(0, dtype=np.float32),
            rL_prefix_min=z,
            rH_prefix_max=z,
            horizon=HN_15m,
            side=side
        )

    # Extract windows
    idx_2d = start[:, None] + np.arange(HN_15m)

    H_win = high_15m[idx_2d]
    L_win = low_15m[idx_2d]
    C_win = close_15m[idx_2d]
    C_end = close_15m[start + HN_15m - 1]"""

if search_build_15m in content:
    content = content.replace(search_build_15m, replace_build_15m)
    print("Replaced build_event_cache_15m part 1")
else:
    print("Could not find build_event_cache_15m part 1")

search_build_15m_2 = """    # Returns
    entry_2d = entry_px[:, None]
    rH = (H_win / np.maximum(entry_2d, eps)) - 1.0
    rL = (L_win / np.maximum(entry_2d, eps)) - 1.0
    rC_end = (C_end / np.maximum(entry_px, eps)) - 1.0"""

replace_build_15m_2 = """    # Returns
    entry_2d = entry_px[:, None]
    rH = (H_win / np.maximum(entry_2d, eps)) - 1.0
    rL = (L_win / np.maximum(entry_2d, eps)) - 1.0
    rC = (C_win / np.maximum(entry_2d, eps)) - 1.0
    rC_end = (C_end / np.maximum(entry_px, eps)) - 1.0"""

if search_build_15m_2 in content:
    content = content.replace(search_build_15m_2, replace_build_15m_2)
    print("Replaced build_event_cache_15m part 2")
else:
    print("Could not find build_event_cache_15m part 2")

search_build_15m_3 = """    return EventCache(
        event_idx=e_1h,  # keep the 1h event index for joining
        entry_px=entry_px,
        rH=rH.astype(np.float32, copy=False),
        rL=rL.astype(np.float32, copy=False),
        rC_end=rC_end.astype(np.float32, copy=False),
        rL_prefix_min=rL_prefix_min,
        rH_prefix_max=rH_prefix_max,
        horizon=HN_15m,
        side=side
    )"""

replace_build_15m_3 = """    return EventCache(
        event_idx=e_1h,  # keep the 1h event index for joining
        entry_px=entry_px,
        rH=rH.astype(np.float32, copy=False),
        rL=rL.astype(np.float32, copy=False),
        rC=rC.astype(np.float32, copy=False),
        rC_end=rC_end.astype(np.float32, copy=False),
        rL_prefix_min=rL_prefix_min,
        rH_prefix_max=rH_prefix_max,
        horizon=HN_15m,
        side=side
    )"""

if search_build_15m_3 in content:
    content = content.replace(search_build_15m_3, replace_build_15m_3)
    print("Replaced build_event_cache_15m part 3")
else:
    print("Could not find build_event_cache_15m part 3")


# Update label_from_cache
search_label = """    # Pessimistic ambiguity resolution: if same bar, SL wins
    ambiguous = (pt_t == sl_t) & (pt_t <= HN - 1)
    pt_first = (pt_t < sl_t)
    sl_first = (sl_t < pt_t)
    # Explicitly pull out ambiguous for diagnostics
    # Note: pt_first and sl_first don't include ambiguous (which is pt_t == sl_t)

    exit_kind = np.zeros(m, dtype=np.int8)
    exit_kind[pt_first] = 1
    # Consistent Ambiguity Resolution (Fix #2, #5, #10):
    # Treat ambiguous (same-bar PT/SL) as SL (-1) for both y_bin and y_ret.
    # We set exit_kind=2 explicitly so we can mask it in diagnostics,
    # but the implementation below treats it as SL.
    exit_kind[sl_first] = -1
    exit_kind[ambiguous] = 2

    # For labels/returns, resolve pessimistically (ambiguous => SL wins)
    sl_pessimistic = sl_first | ambiguous"""

replace_label = """    ambiguous = (pt_t == sl_t) & (pt_t <= HN - 1)
    pt_first = (pt_t < sl_t)
    sl_first = (sl_t < pt_t)

    # Ambiguity resolution: use close price proximity to extreme (high/low depending on trade side)
    ambiguous_pt = np.zeros(m, dtype=bool)
    ambiguous_sl = np.zeros(m, dtype=bool)

    if np.any(ambiguous):
        ambig_idx = np.where(ambiguous)[0]
        ambig_t = pt_t[ambig_idx]

        # We need to extract the corresponding elements from rH, rL, rC
        # We use advanced indexing: cache.rH[ambig_idx, ambig_t]
        rH_ambig = cache.rH[ambig_idx, ambig_t]
        rL_ambig = cache.rL[ambig_idx, ambig_t]
        rC_ambig = cache.rC[ambig_idx, ambig_t]

        dist_to_high = np.abs(rH_ambig - rC_ambig)
        dist_to_low = np.abs(rL_ambig - rC_ambig)

        if cache.side == "long":
            ambig_pt_mask = dist_to_high < dist_to_low
        else:
            ambig_pt_mask = dist_to_low < dist_to_high

        ambig_pt_global_idx = ambig_idx[ambig_pt_mask]
        ambig_sl_global_idx = ambig_idx[~ambig_pt_mask]

        ambiguous_pt[ambig_pt_global_idx] = True
        ambiguous_sl[ambig_sl_global_idx] = True

    # Update pt_first and sl_first with resolved ambiguities
    pt_first = pt_first | ambiguous_pt
    sl_first = sl_first | ambiguous_sl

    exit_kind = np.zeros(m, dtype=np.int8)
    exit_kind[pt_first] = 1
    exit_kind[sl_first] = -1
    # We still keep ambiguous flag for diagnostics if needed, but they are resolved now.
    exit_kind[ambiguous] = 2

    # For labels/returns, we no longer need sl_pessimistic. We use sl_first
    sl_resolved = sl_first"""

if search_label in content:
    content = content.replace(search_label, replace_label)
    print("Replaced label_from_cache part 1")
else:
    print("Could not find label_from_cache part 1")

# Make sure to update the places using sl_pessimistic to use sl_resolved
content = content.replace("y_ret[sl_pessimistic] = -sl_thr[sl_pessimistic]", "y_ret[sl_resolved] = -sl_thr[sl_resolved]")
content = content.replace("time_mask = ~(pt_first | sl_pessimistic)", "time_mask = ~(pt_first | sl_resolved)")


# Also update caching reconstruction logic at line 1162 where we reconstruct `EventCache`
search_cache_reorder = """    cache = EventCache(
        event_idx=e,
        entry_px=cache.entry_px[sort_order],
        rH=cache.rH[sort_order],
        rL=cache.rL[sort_order],
        rC_end=cache.rC_end[sort_order],
        rL_prefix_min=cache.rL_prefix_min[sort_order],
        rH_prefix_max=cache.rH_prefix_max[sort_order],
        horizon=cache.horizon,
        side=cache.side,
    )"""

replace_cache_reorder = """    cache = EventCache(
        event_idx=e,
        entry_px=cache.entry_px[sort_order],
        rH=cache.rH[sort_order],
        rL=cache.rL[sort_order],
        rC=cache.rC[sort_order],
        rC_end=cache.rC_end[sort_order],
        rL_prefix_min=cache.rL_prefix_min[sort_order],
        rH_prefix_max=cache.rH_prefix_max[sort_order],
        horizon=cache.horizon,
        side=cache.side,
    )"""

if search_cache_reorder in content:
    content = content.replace(search_cache_reorder, replace_cache_reorder)
    print("Replaced cache reorder")
else:
    print("Could not find cache reorder")

with open("extreme_price_movements/optimise_tpsl_ratio.py", "w") as f:
    f.write(content)
