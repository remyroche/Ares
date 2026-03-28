import re

with open("extreme_price_movements/training.py", "r") as f:
    content = f.read()

# 2. Update _cluster_geometry_candidates_hybrid
old_cluster = '''def _cluster_geometry_candidates_hybrid(
    triplets,
    ranked_rows,
    archetypes=None,
    topk=None,
    learnability_weight: float = 0.75,
    geometry_weight: float = 0.25,
):
    """Cluster TBM candidates by learnability first, geometry second.

    The candidate universe is the GRR-ranked per-cell pool. The hybrid feature vector is
    dominated by learnability/behavior metrics, with TP/SL geometry descriptors added as a
    smaller share so grouped base models remain economically coherent.
    """
    archetypes = [
        str(a) for a in (archetypes or ["tight", "balanced", "wide"]) if str(a)
    ]
    if not ranked_rows:
        return _split_geometry_triplets_into_archetypes(triplets, archetypes=archetypes)
    ranked_rows = list(ranked_rows)
    if topk is not None:
        try:
            _topk = max(1, int(topk))
            ranked_rows = ranked_rows[:_topk]
        except Exception:
            pass

    triplet_set = {
        (round(float(t[0]), 6), round(float(t[1]), 6), int(t[2])) for t in triplets
    }
    rows = []
    for row in ranked_rows:
        try:
            triplet = (
                round(float(row.get("k_tp")), 6),
                round(float(row.get("sl_as_tp_pct")), 6),
                int(row.get("base_atr_window", 720)),
            )
        except Exception:
            continue
        if triplet not in triplet_set:
            continue
        rows.append((triplet, row))
    if len(rows) < 3:
        return _split_geometry_triplets_into_archetypes(triplets, archetypes=archetypes)

    try:
        from sklearn.cluster import KMeans
    except Exception:
        return _split_geometry_triplets_into_archetypes(triplets, archetypes=archetypes)

    def _safe_num(_row, _key, _default=0.0):
        try:
            _v = float(_row.get(_key, _default))
            return _v if np.isfinite(_v) else float(_default)
        except Exception:
            return float(_default)

    learnability_feats = []
    geometry_feats = []
    widths = []
    kept_triplets = []
    for triplet, row in rows:
        stage2 = _safe_num(row, "stage2_score", 0.0)
        auc_bound = _safe_num(row, "cell_auc_bound", 0.5)
        tp_sep = _safe_num(row, "cell_tp_sep", 0.0)
        bind = _safe_num(row, "cell_bind", 0.0)
        timeout = _safe_num(row, "cell_timeout", 1.0)
        ap_lift = _safe_num(row, "cell_ap_lift", 1.0)
        prod_tp = _safe_num(row, "prod_aligned_tp", 0.0)
        barrier_ratio = _safe_num(
            row, "barrier_ratio", _safe_num(row, "cell_barrier_ratio", 1.0)
        )
        tp_pct = _safe_num(row, "tp_abs_pct", _safe_num(row, "tp_base_pct", 0.0))
        width = float(triplet[0]) + float(triplet[1])
        tp_sl_ratio = float(triplet[0]) / max(float(triplet[1]), 1e-9)
        widths.append(width)
        kept_triplets.append(triplet)
        learnability_feats.append(
            [
                stage2,
                auc_bound,
                tp_sep,
                bind,
                ap_lift,
                prod_tp,
                1.0 - timeout,
            ]
        )
        geometry_feats.append(
            [
                float(triplet[0]),
                float(triplet[1]),
                tp_sl_ratio,
                width,
                barrier_ratio,
                tp_pct,
            ]
        )
    Xl = np.asarray(learnability_feats, dtype=float)
    Xg = np.asarray(geometry_feats, dtype=float)
    if len(kept_triplets) < 3 or np.allclose(np.nanstd(Xl, axis=0), 0.0):
        return _split_geometry_triplets_into_archetypes(
            kept_triplets, archetypes=archetypes
        )

    def _zscore(_x):
        _x = np.nan_to_num(_x, nan=0.0, posinf=0.0, neginf=0.0)
        _mu = np.mean(_x, axis=0, keepdims=True)
        _sd = np.std(_x, axis=0, keepdims=True)
        _sd[_sd < 1e-9] = 1.0
        return (_x - _mu) / _sd

    Xl_z = _zscore(Xl)
    Xg_z = _zscore(Xg)
    lw = float(np.clip(learnability_weight, 0.0, 1.0))
    gw = float(np.clip(geometry_weight, 0.0, 1.0))
    s = lw + gw
    if s <= 1e-9:
        lw, gw = 0.75, 0.25
        s = 1.0
    lw /= s
    gw /= s
    Xz = np.concatenate(
        [np.sqrt(lw) * Xl_z, np.sqrt(gw) * Xg_z],
        axis=1,
    )

    n_clusters = min(3, len(kept_triplets))
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = km.fit_predict(Xz)
    cluster_summary = []
    for cid in range(n_clusters):
        mask = labels == cid
        if not np.any(mask):
            continue
        cluster_summary.append(
            {
                "cid": cid,
                "triplets": [kept_triplets[i] for i in np.where(mask)[0]],
                "stage2_mean": float(
                    np.mean([learnability_feats[i][0] for i in np.where(mask)[0]])
                ),
                "width_mean": float(np.mean([widths[i] for i in np.where(mask)[0]])),
            }
        )
    # Learnability-first ordering: weakest cluster=tight, strongest=wide, middle=balanced.
    # This preserves the "map" semantics while using learnability as the primary grouping axis.
    cluster_summary.sort(key=lambda d: (d["stage2_mean"], d["width_mean"]))
    name_order = ["tight", "balanced", "wide"][-len(cluster_summary) :]
    if len(cluster_summary) == 2:
        name_order = ["tight", "wide"]
    grouped = {}
    for name, info in zip(name_order, cluster_summary):
        if name in archetypes:
            grouped[name] = info["triplets"]
    return grouped or _split_geometry_triplets_into_archetypes(
        kept_triplets, archetypes=archetypes
    )'''

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

content = content.replace(old_cluster, new_cluster)
with open("extreme_price_movements/training.py", "w") as f:
    f.write(content)
