import sys
import re

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# Pattern for bucket_protected_keys logic
p1 = r"""            ranked\["__cheap_rank"\] = cheap_rank
            protected = \(
                ranked\.sort_values\("__cheap_rank", ascending=False\)
                \.head\(max\(min_candidates_per_bucket, 0\)\)\["canonical_key"\]
                \.astype\(str\)
                \.tolist\(\)
            \)
            bucket_protected_keys\[normalized_bucket\] = set\(protected\)"""

r1 = """            ranked["__cheap_rank"] = cheap_rank
            protected = (
                ranked.sort_values("__cheap_rank", ascending=False)
                .head(max(min_candidates_per_bucket, 0))["canonical_key"]
                .astype(str)
                .tolist()
            )
            bucket_protected_keys[normalized_bucket] = set(protected)

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

match1 = re.search(p1, content)
if match1:
    content = content[:match1.start()] + r1 + content[match1.end():]
else:
    print("Could not find match 1")
    sys.exit(1)

# Pattern for subset_auc
p2 = r"""            if not rejected:
                mask_auc, subset_oof_coverage = self\._compute_subset_auc\(
                    X, target_ret, mask, folds
                \)
                if np\.isfinite\(mask_auc\) and np\.isfinite\(global_auc\):
                    lift = mask_auc - global_auc

                mask_entropy = self\._compute_entropy\(target_ret\[mask\]\)
                entropy_red = 1\.0 - \(mask_entropy / \(global_entropy \+ 1e-9\)\)
                if subset_oof_coverage < min_oof_coverage:
                    rejected, rejection_reason = \(
                        True,
                        "insufficient_subset_oof_coverage",
                    \)
                elif not np\.isfinite\(lift\):
                    rejected, rejection_reason = True, "missing_learnability"
                elif lift < 0\.01:  # Lower threshold for lift \(was 1\.10\)
                    rejected, rejection_reason = True, "low_lift\""""

r2 = """            if not rejected:
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
                    mask_auc = np.nan
                    subset_oof_coverage = float(np.mean(mask))
                    lift = 0.0 # Neutral lift
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
                elif run_ridge and lift < 0.01:  # Lower threshold for lift (was 1.10)
                    rejected, rejection_reason = True, "low_lift\""""

match2 = re.search(p2, content)
if match2:
    content = content[:match2.start()] + r2 + content[match2.end():]
else:
    print("Could not find match 2")
    sys.exit(1)

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(content)

print("Patch applied.")
