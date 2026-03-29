with open("extreme_price_movements/training.py", "r") as f:
    content = f.read()

old_cluster = '''def _cluster_geometry_candidates_hybrid(
    triplets,
    ranked_rows,
    archetypes=None,
    topk=None,
    learnability_weight: float = 0.75,
    geometry_weight: float = 0.25,
):
    """Cluster TBM candidates into tight/wide.

    Wide is the ones with wider SL than the median, tight is tighter SL than the median.
    """
    import numpy as np

    archetypes = [
        str(a) for a in (archetypes or ["tight", "wide"]) if str(a)
    ]
    if not triplets:
        return {}

    # We don't really need ranked_rows or KMeans anymore per the new requirement,
    # but we will extract the sl_as_tp_pct from the triplets to partition.

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

    # Group based on sl_as_tp_pct (index 1 of triplet)
    sl_vals = [t[1] for t in uniq_triplets]
    sl_median = np.median(sl_vals)

    tight_group = []
    wide_group = []

    for t in uniq_triplets:
        if t[1] < sl_median:
            tight_group.append(t)
        else:
            wide_group.append(t)

    # Handle edge case where all sl_vals are the same
    if not tight_group:
        # Fall back to splitting by width if sl_as_tp_pct is identical
        mid = len(uniq_triplets) // 2
        tight_group = uniq_triplets[:mid]
        wide_group = uniq_triplets[mid:]
    elif not wide_group:
        mid = len(uniq_triplets) // 2
        wide_group = uniq_triplets[mid:]
        tight_group = uniq_triplets[:mid]

    grouped = {
        "tight": tight_group,
        "wide": wide_group,
    }
    return {k: v for k, v in grouped.items() if k in archetypes and v}'''

new_cluster = '''def _cluster_geometry_candidates_hybrid(
    triplets,
    ranked_rows,
    archetypes=None,
    topk=None,
    learnability_weight: float = 0.75,
    geometry_weight: float = 0.25,
):
    """Cluster TBM candidates into tight/wide.

    Wide is the ones with wider SL than the median, tight is tighter SL than the median.
    If SL is identical to the median, it uses TP (k_tp) as a tie-breaker.
    """
    import numpy as np

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

    # Group based on sl_as_tp_pct (index 1 of triplet)
    sl_vals = [t[1] for t in uniq_triplets]
    sl_median = np.median(sl_vals)

    tight_group = []
    wide_group = []

    # Pre-calculate median of TP (index 0) strictly for those exactly AT the median SL
    # to use as the tie-breaker
    median_sl_triplets = [t for t in uniq_triplets if t[1] == sl_median]
    tp_median = np.median([t[0] for t in median_sl_triplets]) if median_sl_triplets else 0

    for t in uniq_triplets:
        if t[1] < sl_median:
            tight_group.append(t)
        elif t[1] > sl_median:
            wide_group.append(t)
        else:
            # Tie breaker: compare k_tp (index 0) to median k_tp among the tied group
            if t[0] < tp_median:
                tight_group.append(t)
            elif t[0] > tp_median:
                wide_group.append(t)
            else:
                # If both SL and TP are exactly their respective medians,
                # just balance the groups.
                if len(tight_group) <= len(wide_group):
                    tight_group.append(t)
                else:
                    wide_group.append(t)

    # Handle edge case where there is no split even after tie-breaker
    # (e.g. all identical points somehow, though uniq_triplets handles most of that)
    if not tight_group:
        mid = len(uniq_triplets) // 2
        tight_group = uniq_triplets[:mid]
        wide_group = uniq_triplets[mid:]
    elif not wide_group:
        mid = len(uniq_triplets) // 2
        wide_group = uniq_triplets[mid:]
        tight_group = uniq_triplets[:mid]

    grouped = {
        "tight": tight_group,
        "wide": wide_group,
    }
    return {k: v for k, v in grouped.items() if k in archetypes and v}'''

content = content.replace(old_cluster, new_cluster)
with open("extreme_price_movements/training.py", "w") as f:
    f.write(content)
