with open("extreme_price_movements/elasticnet_feature_selection_v2.py", "r") as f:
    code = f.read()

# Update Path Building Metrics inside the grid search loop
old_metrics = """            path_results.append({
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "mean_score": float(np.mean(fold_scores)),
                "std_score": float(np.std(fold_scores)),
                "n_selected": n_sel,
                "selected_union": union_list,
                "mean_sel_freq": mean_sel_freq,
                "mean_jaccard": mean_jaccard,
                "stability_score": stab_score,
                "freqs": freqs,
                "path_zero_fold_count": all_zero_count,
                "selection_freq_min": sel_freq_min,
                "selection_freq_mean": mean_sel_freq,
                "selection_freq_max": sel_freq_max,
                "mean_sign_consistency": msc
            })"""

new_metrics = """            norm_n = n_sel / max(1, n_features)
            adj_score = stab_score - sparsity_penalty_eff * norm_n

            path_results.append({
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "mean_score": float(np.mean(fold_scores)),
                "std_score": float(np.std(fold_scores)),
                "n_selected": n_sel,
                "selected_union": union_list,
                "mean_sel_freq": mean_sel_freq,
                "mean_jaccard": mean_jaccard,
                "stability_score": stab_score,
                "freqs": freqs,
                "path_zero_fold_count": all_zero_count,
                "selection_freq_min": sel_freq_min,
                "selection_freq_mean": mean_sel_freq,
                "selection_freq_max": sel_freq_max,
                "mean_sign_consistency": msc,
                "normalized_n_selected": norm_n,
                "adjusted_score": adj_score
            })"""

code = code.replace(old_metrics, new_metrics)

with open("extreme_price_movements/elasticnet_feature_selection_v2.py", "w") as f:
    f.write(code)
