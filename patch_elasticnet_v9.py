with open("extreme_price_movements/elasticnet_feature_selection_v2.py", "r") as f:
    code = f.read()

# Update the empty-path fallback
old_empty = """    if not path_results:
        return {
            "selected_idx": np.arange(n_features),
            "selected_names": feature_names,
            "best_alpha": 0.0, "best_l1_ratio": 0.0,
            "chosen_alpha": 0.0, "chosen_l1_ratio": 0.0,
            "best_score": 0.0, "score_std": 0.0, "threshold_score": 0.0,
            "n_features_selected": n_features,
            "selection_frequency": np.ones(n_features),
            "mean_jaccard": 1.0,
            "mean_sign_consistency": None,
            "stability_score": 1.0,
            "adjusted_score": 1.0,
            "effective_selection_freq_threshold": selection_freq_threshold,
            "model_feature_floor": min_features_floor,
            "was_backfilled_to_floor": False,
            "path_table": pd.DataFrame()
        }"""

new_empty = """    if not path_results:
        fallback_n = min(n_features, max(1, min_features_floor))
        fallback_idx = np.arange(fallback_n, dtype=int)
        fallback_names = [feature_names[i] for i in fallback_idx]

        return {
            "selected_idx": fallback_idx,
            "selected_names": fallback_names,
            "best_alpha": 0.0,
            "best_l1_ratio": 0.0,
            "chosen_alpha": 0.0,
            "chosen_l1_ratio": 0.0,
            "best_score": 0.0,
            "score_std": 0.0,
            "threshold_score": 0.0,
            "n_features_selected": len(fallback_idx),
            "selection_frequency": np.zeros(n_features),
            "selected_feature_frequencies": np.zeros(len(fallback_idx)),
            "mean_jaccard": 0.0,
            "mean_sign_consistency": None,
            "stability_score": 0.0,
            "adjusted_score": 0.0,
            "effective_selection_freq_threshold": eff_sel_thresh,
            "model_feature_floor": target_floor,
            "was_backfilled_to_floor": True,
            "path_table": pd.DataFrame(),
        }"""

code = code.replace(old_empty, new_empty)

with open("extreme_price_movements/elasticnet_feature_selection_v2.py", "w") as f:
    f.write(code)
