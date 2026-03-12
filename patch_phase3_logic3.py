import re

with open("extreme_price_movements/mask_optimiser.py", "r") as f:
    code = f.read()

# Wait, `df_short` already goes through `_compute_phase3_feature_learnability` inside `for _, row in df2.iterrows():`
# which is before `df_short` is created. `df_short` is a subset of `df2`.
# Let's check where `df2 = df2.sort_values... df_short = df2.head...` happens.
