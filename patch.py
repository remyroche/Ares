import re

with open("extreme_price_movements/simple_position_sizer.py", "r") as f:
    content = f.read()

# 1. Update clean_and_standardize signature and logic
old_clean = """def clean_and_standardize(X: np.ndarray, fit_medians: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    \"\"\"Standardizes features safely handling NaNs and Infs.\"\"\"
    X_clean = X.copy()
    X_clean[np.isinf(X_clean)] = np.nan

    if fit_medians is None:
        fit_medians = np.nanmedian(X_clean, axis=0)
        fit_medians[np.isnan(fit_medians)] = 0.0

    if X_clean.ndim == 1:
        inds = np.isnan(X_clean)
        X_clean[inds] = fit_medians
        # Simple standard scaling for 1D
        std = np.std(X_clean)
        if std > 1e-9:
            X_clean = (X_clean - np.mean(X_clean)) / std
        else:
            X_clean = X_clean - np.mean(X_clean)
    else:
        inds = np.where(np.isnan(X_clean))
        X_clean[inds] = np.take(fit_medians, inds[1])
        scaler = StandardScaler()
        X_clean = scaler.fit_transform(X_clean)

    return X_clean, fit_medians"""

new_clean = """def clean_and_standardize(X: np.ndarray, fit_medians: Optional[np.ndarray] = None, scaler: Optional[StandardScaler] = None, mean_1d: Optional[float] = None, std_1d: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Any, Any, Any]:
    \"\"\"Standardizes features safely handling NaNs and Infs.\"\"\"
    X_clean = X.copy()
    X_clean[np.isinf(X_clean)] = np.nan

    if fit_medians is None:
        fit_medians = np.nanmedian(X_clean, axis=0)
        if np.isscalar(fit_medians):
            if np.isnan(fit_medians):
                fit_medians = 0.0
        else:
            fit_medians[np.isnan(fit_medians)] = 0.0

    if X_clean.ndim == 1:
        inds = np.isnan(X_clean)
        X_clean[inds] = fit_medians

        if mean_1d is None or std_1d is None:
            mean_1d = np.mean(X_clean)
            std_1d = np.std(X_clean)

        if std_1d > 1e-9:
            X_clean = (X_clean - mean_1d) / std_1d
        else:
            X_clean = X_clean - mean_1d
    else:
        inds = np.where(np.isnan(X_clean))
        X_clean[inds] = np.take(fit_medians, inds[1])

        if scaler is None:
            scaler = StandardScaler()
            X_clean = scaler.fit_transform(X_clean)
        else:
            X_clean = scaler.transform(X_clean)

    return X_clean, fit_medians, scaler, mean_1d, std_1d"""
content = content.replace(old_clean, new_clean)


# 2. Update SimpleHeadRidgeSizer scaling
old_ridge = """            # Fold-local scaling and NaN cleaning
            X_tr_clean, medians = clean_and_standardize(X_tr)
            X_te_clean, _ = clean_and_standardize(X_te, fit_medians=medians)"""

new_ridge = """            # Fold-local scaling and NaN cleaning
            X_tr_clean, medians, scaler, mean_1d, std_1d = clean_and_standardize(X_tr)
            X_te_clean, _, _, _, _ = clean_and_standardize(X_te, fit_medians=medians, scaler=scaler, mean_1d=mean_1d, std_1d=std_1d)"""
content = content.replace(old_ridge, new_ridge)


# 3. Update build_combo_candidates _norm
old_norm = """    def _norm(x):
        x_c, _ = clean_and_standardize(x)
        return x_c"""

new_norm = """    def _norm(x):
        x_c, _, _, _, _ = clean_and_standardize(x)
        return x_c"""
content = content.replace(old_norm, new_norm)


# 4. Update run_stage_2_combo_race signature
old_race_sig = """def run_stage_2_combo_race(
    candidates: Dict[str, np.ndarray],
    y_raw_net_return: np.ndarray,
    y_downside: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, Any]]:"""

new_race_sig = """def run_stage_2_combo_race(
    candidates: Dict[str, np.ndarray],
    y_raw_net_return: np.ndarray,
    y_downside: np.ndarray,
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:"""
content = content.replace(old_race_sig, new_race_sig)


# 5. Update run_stage_2_combo_race evaluation logic
old_race_eval = """        # Combos are pre-aligned so higher score = better expected outcome.
        # We pass directionality "return-like" because we built them that way.
        metrics = evaluate_signal(name, scores, y_raw_net_return, y_downside, directionality="return-like")
        # Rename head_name to combo_name for clarity
        metrics["combo_name"] = metrics.pop("head_name")
        results.append(metrics)"""

new_race_eval = """        # Combos are pre-aligned so higher score = better expected outcome.
        # We pass directionality "return-like" because we built them that way.
        metrics = evaluate_signal(name, scores, y_raw_net_return, y_downside, directionality="return-like")

        # Calculate fold-level stability
        if splits:
            fold_spearmans = []
            for tr_idx, te_idx in splits:
                if len(te_idx) > 0:
                    corr, _ = spearmanr(scores[te_idx], y_raw_net_return[te_idx], nan_policy="omit")
                    if pd.notna(corr):
                        fold_spearmans.append(float(corr))
            if fold_spearmans:
                metrics["fold_spearman_mean"] = float(np.mean(fold_spearmans))
                metrics["fold_spearman_std"] = float(np.std(fold_spearmans))
            else:
                metrics["fold_spearman_mean"] = 0.0
                metrics["fold_spearman_std"] = 0.0

        # Rename head_name to combo_name for clarity
        metrics["combo_name"] = metrics.pop("head_name")
        results.append(metrics)"""
content = content.replace(old_race_eval, new_race_eval)


# 6. Update run_simple_position_sizer to use splits
old_main_race = """    # 3. Stage 2 Combo Race
    combo_candidates = build_combo_candidates(feature_dict, detected_heads, lambda_grid)
    stage_2_df, best_combo = run_stage_2_combo_race(combo_candidates, y_raw_net_return, y_downside)"""

new_main_race = """    # Determine temporal splits for stability checking and OOF Ridge
    n_samples = len(y_raw_net_return)
    splits = simple_temporal_splits(timestamps, n_samples)

    # 3. Stage 2 Combo Race
    combo_candidates = build_combo_candidates(feature_dict, detected_heads, lambda_grid)
    stage_2_df, best_combo = run_stage_2_combo_race(combo_candidates, y_raw_net_return, y_downside, splits)"""
content = content.replace(old_main_race, new_main_race)


# 7. Update run_simple_position_sizer Ridge initialization
old_main_ridge = """        splits = simple_temporal_splits(timestamps, n_samples)

        ridge = SimpleHeadRidgeSizer(alpha=1.0)"""
new_main_ridge = """        ridge = SimpleHeadRidgeSizer(alpha=1.0)"""
content = content.replace(old_main_ridge, new_main_ridge)


# 8. Update run_simple_position_sizer best check
old_main_best = """        # Compare Ridge vs Best Combo
        if best_combo:
            if ridge_metrics.get("utility_score", 0) > best_combo.get("utility_score", 0):
                best_simple_score = ridge_oof_preds
                best_simple_score_name = "Ridge_Head_Sizer\""""
new_main_best = """        # Compare Ridge vs Best Combo
        if not best_combo or ridge_metrics.get("utility_score", 0) > best_combo.get("utility_score", -9999):
            best_simple_score = ridge_oof_preds
            best_simple_score_name = "Ridge_Head_Sizer\""""
content = content.replace(old_main_best, new_main_best)

with open("extreme_price_movements/simple_position_sizer.py", "w") as f:
    f.write(content)
