import re
with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# Make sure we store the rejection reasons correctly
# In assess_rules:
# if ev_per_event <= 0:
#     rejected, rejection_reason = True, "ev_per_event <= 0"
# Wait, rejection_data was set for threshold star failure but not passed into assessment_results

pattern = r'"rejection_reason": rejection_reason,'

new_pattern = r'"rejection_reason": rejection_reason,\n                    "rejection_data": rejection_data if "rejection_data" in locals() else None,'

source = source.replace(pattern, new_pattern)

with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
    f.write(source)
    print("Patched rejection data")
