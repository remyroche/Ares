import re

with open("extreme_price_movements/training.py", "r") as f:
    content = f.read()

# 1. Update _split_geometry_triplets_into_archetypes
old_split = '''def _split_geometry_triplets_into_archetypes(triplets, archetypes=None):
    """Partition validated TBM triplets into tight/balanced/wide archetypes.

    The split is deterministic and based on joint TP/SL geometry width. Lower-width
    configurations are `tight`, higher-width are `wide`, and the middle band is
    `balanced`. When too few distinct triplets exist, only the available groups are
    returned and callers should keep the canonical aggregate as the primary label set.
    """
    archetypes = [
        str(a) for a in (archetypes or ["tight", "balanced", "wide"]) if str(a)
    ]
    if not triplets:
        return {}
    uniq_triplets = []
    seen = set()
    for t in triplets:
        key = (round(float(t[0]), 6), round(float(t[1]), 6), int(t[2]))
        if key in seen:
            continue
        seen.add(key)
        uniq_triplets.append((float(t[0]), float(t[1]), int(t[2])))
    if len(uniq_triplets) <= 1:
        return {"balanced": uniq_triplets}

    scored = []
    for t in uniq_triplets:
        k_tp, sl_tp, atr_win = t
        width_score = float(k_tp) + float(sl_tp)
        asym_score = abs(float(k_tp) - float(sl_tp))
        scored.append((width_score, asym_score, atr_win, t))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    ordered = [x[-1] for x in scored]
    n = len(ordered)
    if n == 2:
        return {"tight": [ordered[0]], "wide": [ordered[1]]}

    idx_tight_end = max(1, int(np.floor(n / 3)))
    idx_wide_start = min(n - 1, int(np.ceil(2 * n / 3)))
    groups = {
        "tight": ordered[:idx_tight_end],
        "balanced": ordered[idx_tight_end:idx_wide_start],
        "wide": ordered[idx_wide_start:],
    }
    # Fold empty middle buckets back into balanced using the closest remaining rows.
    if not groups["balanced"]:
        mid = n // 2
        groups["balanced"] = [ordered[mid]]
        groups["tight"] = ordered[:mid]
        groups["wide"] = ordered[mid + 1 :]
    return {k: v for k, v in groups.items() if k in archetypes and v}'''

new_split = '''def _split_geometry_triplets_into_archetypes(triplets, archetypes=None):
    """Partition validated TBM triplets into tight/wide archetypes.

    The split is deterministic and based on joint TP/SL geometry width. Lower-width
    configurations are `tight`, higher-width are `wide`.
    """
    archetypes = [
        str(a) for a in (archetypes or ["tight", "wide"]) if str(a)
    ]
    if not triplets:
        return {}
    uniq_triplets = []
    seen = set()
    for t in triplets:
        key = (round(float(t[0]), 6), round(float(t[1]), 6), int(t[2]))
        if key in seen:
            continue
        seen.add(key)
        uniq_triplets.append((float(t[0]), float(t[1]), int(t[2])))
    if len(uniq_triplets) <= 1:
        return {"tight": uniq_triplets}

    scored = []
    for t in uniq_triplets:
        k_tp, sl_tp, atr_win = t
        width_score = float(k_tp) + float(sl_tp)
        asym_score = abs(float(k_tp) - float(sl_tp))
        scored.append((width_score, asym_score, atr_win, t))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    ordered = [x[-1] for x in scored]
    n = len(ordered)

    mid = n // 2
    groups = {
        "tight": ordered[:mid],
        "wide": ordered[mid:],
    }
    return {k: v for k, v in groups.items() if k in archetypes and v}'''

content = content.replace(old_split, new_split)
with open("extreme_price_movements/training.py", "w") as f:
    f.write(content)
