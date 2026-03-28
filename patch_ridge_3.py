import sys
import re

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# Instead of modifying `MaskAssessor.assess_rules`, we can look at where `_compute_subset_auc` is called inside `assess_rules`.
# It's at:
#             # 7. Learnability (Efficiency Frontier) - expensive section
#             global_auc = global_auc_by_side[side]
#             global_entropy = global_entropy_by_side[side]
#             subset_oof_coverage = 0.0
#             lift = np.nan
#             entropy_red = np.nan
#             if not rejected:
#                 mask_auc, subset_oof_coverage = self._compute_subset_auc(
#                     X, target_ret, mask, folds
#                 )
#                 ...
# This IS where the Ridge scoring happens. It's called for every rule that passed the cheap gates!
# The user wants to "ensure we run Ridge models on a selected subset only; maybe Ridge on top20 per target x side x horizon? using what metrics for the decisions?"
# So, we should select the top 20 candidates per bucket before evaluating Ridge!
# Let's see how `assess_rules` operates. It loops over `registry.iterrows()`.
# To do this efficiently, we can first compute all the "cheap" stats for all rules, filter out the rejected ones, select the top 20 by some metric (like `regime_score` without lift, or `cheap_rank`), and THEN compute the expensive Ridge metric `_compute_subset_auc` ONLY for those 20 rules.

# Wait, `assess_rules` is processing all rules in `registry`.
# It already computes a `cheap_rank`:
#             cheap_rank = (
#                 pd.to_numeric(group_df["directional_mean_ret"], errors="coerce")
#                 .fillna(0.0)
#                 .to_numpy(dtype=float)
#                 + pd.to_numeric(group_df["trade_path_quality_score"], errors="coerce")
#                 .fillna(0.0)
#                 .to_numpy(dtype=float)
#                 + pd.to_numeric(group_df["quality_stability_score"], errors="coerce")
#                 .fillna(0.0)
#                 .to_numpy(dtype=float)
#             )
# And it already identifies the top candidates per bucket:
#             protected = (
#                 ranked.sort_values("__cheap_rank", ascending=False)
#                 .head(max(min_candidates_per_bucket, 0))["canonical_key"]
#                 .astype(str)
#                 .tolist()
#             )
#             bucket_protected_keys[normalized_bucket] = set(protected)
# Here `min_candidates_per_bucket` is `int(self.cfg.get("min_candidates_per_bucket", 50))`.
# Oh! So it already kind of protects 50 rules per bucket.

# But the user says: "ensure we run Ridge models on a selected subset only; maybe Ridge on top20 per target x side x horizon? using what metrics for the decisions?"
# Okay, so I should modify the logic so that `_compute_subset_auc` is ONLY run for the top 20 rules (by `__cheap_rank`) per target x side x horizon. For the rest of the rules (if any pass), `lift` is skipped or they are rejected.
# Even better: we can add a new configuration parameter `max_ridge_candidates_per_bucket` (default 20) and only run `_compute_subset_auc` if the rule is in the top 20 of `__cheap_rank` for its bucket. If it's not in the top 20, we don't run Ridge and either assume `lift = 0` or reject it due to "not_in_top_ridge_candidates".

print(content[6750:6850])
