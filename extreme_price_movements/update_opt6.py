import re

with open("extreme_price_movements/optimise_tpsl_ratio.py", "r") as f:
    content = f.read()

search_build3 = """    return EventCache(
        event_idx=e.astype(np.int32, copy=False),
        entry_px=entry_px.astype(np.float32, copy=False),
        rH=rH.astype(np.float32, copy=False),
        rL=rL.astype(np.float32, copy=False),
        rC_end=rC_end.astype(np.float32, copy=False),
        rL_prefix_min=rL_prefix_min,
        rH_prefix_max=rH_prefix_max,
        horizon=HN,
        side=side
    )"""

replace_build3 = """    return EventCache(
        event_idx=e.astype(np.int32, copy=False),
        entry_px=entry_px.astype(np.float32, copy=False),
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

    offs = np.arange(HN_15m, dtype=np.int32)[None, :]
    widx = start[:, None] + offs

    H = high_15m[widx]
    L = low_15m[widx]
    C_end = close_15m[widx[:, -1]]

    denom = np.maximum(entry_px, eps).astype(np.float32, copy=False)
    rH = (H / denom[:, None]) - 1.0
    rL = (L / denom[:, None]) - 1.0
    rC_end = (C_end / denom) - 1.0"""

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

    offs = np.arange(HN_15m, dtype=np.int32)[None, :]
    widx = start[:, None] + offs

    H = high_15m[widx]
    L = low_15m[widx]
    C = close_15m[widx]
    C_end = close_15m[widx[:, -1]]

    denom = np.maximum(entry_px, eps).astype(np.float32, copy=False)
    rH = (H / denom[:, None]) - 1.0
    rL = (L / denom[:, None]) - 1.0
    rC = (C / denom[:, None]) - 1.0
    rC_end = (C_end / denom) - 1.0"""

if search_build_15m in content:
    content = content.replace(search_build_15m, replace_build_15m)
    print("Replaced build_event_cache_15m part 1")
else:
    print("Could not find build_event_cache_15m part 1")

search_build_15m_3 = """    return EventCache(
        event_idx=e_1h.astype(np.int32, copy=False),  # keep the 1h event index for joining
        entry_px=entry_px.astype(np.float32, copy=False),
        rH=rH.astype(np.float32, copy=False),
        rL=rL.astype(np.float32, copy=False),
        rC_end=rC_end.astype(np.float32, copy=False),
        rL_prefix_min=rL_prefix_min,
        rH_prefix_max=rH_prefix_max,
        horizon=HN_15m,
        side=side
    )"""

replace_build_15m_3 = """    return EventCache(
        event_idx=e_1h.astype(np.int32, copy=False),  # keep the 1h event index for joining
        entry_px=entry_px.astype(np.float32, copy=False),
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


with open("extreme_price_movements/optimise_tpsl_ratio.py", "w") as f:
    f.write(content)
