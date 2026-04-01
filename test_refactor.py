import re
with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# Check for hard filters on ret_uplift or cheap_rank
# The prompt says: Remove cheap_rank and ret_uplift from final-stage scoring.
# Remove the old ret_uplift hard filter.
# Let's see if there are any hard filters on ret_uplift in assess_rules
# It might be in the pre-filter part or in assess_rules itself.

matches = re.finditer(r"ret_uplift.*?rejected", source, re.DOTALL)
for m in matches:
    print(m.group(0)[:100])
