with open("extreme_price_movements/elasticnet_feature_selection_v2.py", "r") as f:
    code = f.read()

old_inner = """def _compute_inner_splits(timestamps: Optional[np.ndarray], n_samples: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_samples < max(20, n_splits * 5):
        mid = n_samples // 2
        return [(np.arange(0, mid), np.arange(mid, n_samples))]

    cv = PurgedKFold(n_splits=n_splits, purge=43200, embargo=43200, times=timestamps)
    dummy_X = np.empty((n_samples, 1))
    splits = []
    for tr, va in cv.split(dummy_X):
        if len(tr) > 0 and len(va) > 0:
            splits.append((tr, va))

    if not splits:
        mid = n_samples // 2
        return [(np.arange(0, mid), np.arange(mid, n_samples))]
    return splits"""

new_inner = """def _compute_inner_splits(timestamps: Optional[np.ndarray], n_samples: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_samples < max(20, n_splits * 5):
        mid = max(1, n_samples // 2)
        if mid >= n_samples:
            return []
        return [(np.arange(0, mid), np.arange(mid, n_samples))]

    cv = PurgedKFold(n_splits=n_splits, purge=43200, embargo=43200, times=timestamps)
    dummy_X = np.empty((n_samples, 1))
    splits = []
    for tr, va in cv.split(dummy_X):
        if len(tr) > 0 and len(va) > 0:
            splits.append((tr, va))

    if not splits:
        mid = max(1, n_samples // 2)
        if mid >= n_samples:
            return []
        return [(np.arange(0, mid), np.arange(mid, n_samples))]
    return splits"""

code = code.replace(old_inner, new_inner)

with open("extreme_price_movements/elasticnet_feature_selection_v2.py", "w") as f:
    f.write(code)
