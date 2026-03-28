import sys
import re

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# Replace `protected = (` with our new logic for top 20 Ridge candidates.
# Wait, actually we can just introduce `max_ridge_candidates_per_bucket` and `bucket_ridge_keys`.
# Around line 6625:
#             ranked["__cheap_rank"] = cheap_rank
#             protected = (
#                 ranked.sort_values("__cheap_rank", ascending=False)
#                 .head(max(min_candidates_per_bucket, 0))["canonical_key"]
#                 .astype(str)
#                 .tolist()
#             )
#             bucket_protected_keys[normalized_bucket] = set(protected)

new_code = """            ranked["__cheap_rank"] = cheap_rank
            protected = (
                ranked.sort_values("__cheap_rank", ascending=False)
                .head(max(min_candidates_per_bucket, 0))["canonical_key"]
                .astype(str)
                .tolist()
            )
            bucket_protected_keys[normalized_bucket] = set(protected)

            # Select top 20 candidates for Ridge scoring based on cheap metrics
            max_ridge_candidates_per_bucket = int(self.cfg.get("max_ridge_candidates_per_bucket", 20))
            ridge_cands = (
                ranked.sort_values("__cheap_rank", ascending=False)
                .head(max(max_ridge_candidates_per_bucket, 0))["canonical_key"]
                .astype(str)
                .tolist()
            )
            if not hasattr(self, 'bucket_ridge_keys'):
                self.bucket_ridge_keys = {}
            self.bucket_ridge_keys[normalized_bucket] = set(ridge_cands)"""

pattern = r'            ranked\["__cheap_rank"\] = cheap_rank\n            protected = \(\n                ranked\.sort_values\("__cheap_rank", ascending=False\)\n                \.head\(max\(min_candidates_per_bucket, 0\)\)\["canonical_key"\]\n                \.astype\(str\)\n                \.tolist\(\)\n            \)\n            bucket_protected_keys\[normalized_bucket\] = set\(protected\)'

match = re.search(pattern, content)
if not match:
    print("Could not find protected logic")
    sys.exit(1)

content = content[:match.start()] + new_code + content[match.end():]

# Now, we also need to change where `_compute_subset_auc` is called.
# Around line 6813:
#             if not rejected:
#                 mask_auc, subset_oof_coverage = self._compute_subset_auc(
#                     X, target_ret, mask, folds
#                 )
#                 if np.isfinite(mask_auc) and np.isfinite(global_auc):
#                     lift = mask_auc - global_auc

new_code_2 = """            if not rejected:
                # Only run expensive Ridge scoring if this rule is in the top N Ridge candidates for its bucket
                # (determined earlier by __cheap_rank)
                run_ridge = False
                if hasattr(self, 'bucket_ridge_keys') and bucket_key in self.bucket_ridge_keys:
                    if canonical_key in self.bucket_ridge_keys[bucket_key]:
                        run_ridge = True

                if run_ridge:
                    mask_auc, subset_oof_coverage = self._compute_subset_auc(
                        X, target_ret, mask, folds
                    )
                    if np.isfinite(mask_auc) and np.isfinite(global_auc):
                        lift = mask_auc - global_auc
                else:
                    # For rules not in top Ridge candidates, assume a neutral lift or reject
                    mask_auc = np.nan
                    subset_oof_coverage = float(np.mean(mask))
                    lift = 0.0 # Neutral lift so it doesn't penalize regime score too much if we keep it, but it shouldn't be high enough to win
                    rejected, rejection_reason = True, "not_in_top_ridge_candidates"

                mask_entropy = self._compute_entropy(target_ret[mask])
                entropy_red = 1.0 - (mask_entropy / (global_entropy + 1e-9))
                if subset_oof_coverage < min_oof_coverage:
                    rejected, rejection_reason = (
                        True,
                        "insufficient_subset_oof_coverage",
                    )
                elif not np.isfinite(lift):
                    rejected, rejection_reason = True, "missing_learnability"
                elif lift < 0.01 and run_ridge:  # Lower threshold for lift
                    rejected, rejection_reason = True, "low_lift\"\"\"

# Actually, I should match exactly what is there. Let's see it.
"""
with open('patch_ridge_4.py', 'w') as f: f.write("")
