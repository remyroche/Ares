import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

source = source.replace(
    '"rejection_reason": rejection_reason,\\n                    "rejection_data": rejection_data if "rejection_data" in locals() else None,',
    '"rejection_reason": rejection_reason,\n                    "rejection_data": rejection_data if "rejection_data" in locals() else None,'
)

with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
    f.write(source)
