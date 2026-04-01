import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

pattern = r"def _compute_subset_auc\(self, X, fwd_ret, mask, folds\)(.*?)return self._compute_oof_learnability_score\((.*?)\)"
match = re.search(pattern, source, re.DOTALL)
if match:
    print("Found _compute_subset_auc")
else:
    print("Could not find _compute_subset_auc")
