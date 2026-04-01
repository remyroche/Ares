import re

with open("tests/test_lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

source = source.replace(
    "auc, coverage = assessor._compute_subset_auc(x, fwd_ret, mask, folds)",
    "auc, coverage, _ = assessor._compute_subset_auc(x, fwd_ret, mask, folds)"
)

with open("tests/test_lgbm_based_mask_generation.py", "w") as f:
    f.write(source)
