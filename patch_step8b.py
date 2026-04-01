with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# `select_top_diverse_rules` is called in `run_mining_stage`.
# The current usage looks like:
# top_diverse = select_top_diverse_rules(
#     assessment_df, mask_map, top_n=15, max_overlap=0.4, max_side_in_top=9
# )

import re
old_call = re.compile(r"select_top_diverse_rules\(\s*.*?,\s*.*?,\s*top_n=.*?max_side_in_top=.*?\)", re.DOTALL)

# Let's replace it manually to be safe.
# Find the line in run_mining_stage
call1 = """    # Select top 15 diverse rules
    top_diverse = select_top_diverse_rules(
        assessment_df, mask_map, top_n=15, max_overlap=0.4, max_side_in_top=9
    )"""

new_call1 = """    # Select top 10 diverse rules using greedy algorithm
    top_diverse = select_final_regimes(
        assessment_df, mask_map, top_n=10
    )"""

if call1 in source:
    source = source.replace(call1, new_call1)
    with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
        f.write(source)
    print("Replaced call successfully.")
else:
    print("Could not find exact string. Searching via regex.")
