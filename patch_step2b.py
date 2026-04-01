with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

old_signature = """def _compute_subset_auc(self, X, fwd_ret, mask, folds) -> Tuple[float, float]:"""
new_signature = """def _compute_subset_auc(self, X, fwd_ret, mask, folds) -> Tuple[float, float, np.ndarray]:"""

old_return = """        return self._compute_oof_learnability_score(
            oof_preds, y, mask, min_predicted_points=min_pred_points
        )"""

new_return = """        score, coverage = self._compute_oof_learnability_score(
            oof_preds, y, mask, min_predicted_points=min_pred_points
        )
        return score, coverage, oof_preds"""

if old_signature in source and old_return in source:
    source = source.replace(old_signature, new_signature)
    source = source.replace(old_return, new_return)

    # Now find the call site
    old_call = """mask_auc, subset_oof_coverage = self._compute_subset_auc(
                        X, target_ret, mask, folds
                    )"""
    new_call = """mask_auc, subset_oof_coverage, oof_preds = self._compute_subset_auc(
                        X, target_ret, mask, folds
                    )"""

    if old_call in source:
        source = source.replace(old_call, new_call)
        with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
            f.write(source)
        print("Patched successfully")
    else:
        print("Could not find call site")
else:
    print("Could not find signature or return")
