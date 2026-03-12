import re

with open("extreme_price_movements/mask_optimiser.py", "r") as f:
    code = f.read()

# I notice that `new_metrics` contains conditional_predictability_gain etc. but we also need TBM and other metrics.
# Right now Phase 3 is spread. The `eval_candidate` uses `_compute_full_metrics_for_candidate` which is just Phase 2 stuff.
# Phase 3 actually needs feature_learnability, conditional_predictability, TBM economic, and Phase 4 TBM LGBM metrics.
# Those are currently computed inside the later loop:
