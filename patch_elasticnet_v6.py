with open("extreme_price_movements/elasticnet_feature_selection_v2.py", "r") as f:
    code = f.read()

# Update signature
old_sig = """def select_features_via_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    timestamps_train: Optional[np.ndarray],
    model_kind: str,
    feature_names: List[str],
    alpha_grid: np.ndarray,
    l1_ratio_grid: List[float],
    sample_weight_train: Optional[np.ndarray] = None,
    inner_n_splits: int = 2,
    max_features_cap: Optional[int] = None,
    min_features_floor: int = 5,
    sparsity_penalty: float = 0.04,
    selection_freq_threshold: float = 0.67,
    use_sign_consistency: bool = False,
) -> Dict:"""

new_sig = """def select_features_via_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    timestamps_train: Optional[np.ndarray],
    model_kind: str,
    feature_names: List[str],
    alpha_grid: np.ndarray,
    l1_ratio_grid: List[float],
    sample_weight_train: Optional[np.ndarray] = None,
    inner_n_splits: int = 3,
    max_features_cap: Optional[int] = None,
    min_features_floor: int = 5,
    sparsity_penalty: float = 0.04,
    selection_freq_threshold: float = 0.67,
    use_sign_consistency: bool = False,
) -> Dict:"""

code = code.replace(old_sig, new_sig)


# Initial calculations block
old_init = """    n_samples, n_features = X_train.shape
    if max_features_cap is None:
        max_features_cap = n_features

    splits = _compute_inner_splits(timestamps_train, n_samples, inner_n_splits)
    if not splits:
        # Fallback block split
        mid = n_samples // 2
        splits = [(np.arange(0, mid), np.arange(mid, n_samples))]"""

new_init = """    n_samples, n_features = X_train.shape
    if max_features_cap is None:
        max_features_cap = n_features

    target_floor = min(min_features_floor, max_features_cap, n_features)

    if model_kind == "edge":
        sparsity_penalty_eff = 0.04
    elif model_kind == "downside":
        sparsity_penalty_eff = 0.035
    elif model_kind == "uncertainty":
        sparsity_penalty_eff = 0.025
    else:
        sparsity_penalty_eff = sparsity_penalty

    if inner_n_splits <= 2:
        eff_sel_thresh = 0.50
    elif inner_n_splits == 3:
        eff_sel_thresh = 2.0 / 3.0
    else:
        eff_sel_thresh = selection_freq_threshold

    splits = _compute_inner_splits(timestamps_train, n_samples, inner_n_splits)
    if not splits:
        mid = max(1, n_samples // 2)
        if mid < n_samples:
            splits = [(np.arange(0, mid), np.arange(mid, n_samples))]"""

code = code.replace(old_init, new_init)

with open("extreme_price_movements/elasticnet_feature_selection_v2.py", "w") as f:
    f.write(code)
