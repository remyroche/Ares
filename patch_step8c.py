import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# I will replace `select_top_diverse_rules` with `select_final_regimes` where it is called.
# It's called in main() or in `run_mining_stage`. Let's see the first hit:
# `top_final = select_top_diverse_rules(combined_global_registry, combined_mask_map, top_n=15)`
# It should be `top_final = select_final_regimes(combined_global_registry, combined_mask_map, top_n=10)`

# Note the user said: "keep the top 10 candidates using greedy, order-dependent selection... accept/reject ... keep the final top 10 accepted candidates".
# I'll modify the call at line 5917.

pattern1 = r"top_final = select_top_diverse_rules\(\s*combined_global_registry,\s*combined_mask_map,\s*top_n=15\s*\)"
new_pattern1 = r"top_final = select_final_regimes(combined_global_registry, combined_mask_map, top_n=10)"

source = re.sub(pattern1, new_pattern1, source)

# Are there any other calls? Let's search again.
# One is the recursive call inside select_top_diverse_rules, which doesn't matter since we won't use it.
# Any other?

with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
    f.write(source)

print("Patched calls to select_top_diverse_rules.")
