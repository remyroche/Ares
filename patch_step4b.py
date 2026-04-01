import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

pattern = r"(        assessment_results = \[\]\n)"
match = re.search(pattern, source)
if match:
    source = source[:match.end(1)] + "\n        total_symbol_days = self._compute_total_symbol_days(data)\n        if total_symbol_days is None:\n            total_symbol_days = 1.0\n" + source[match.end(1):]
    with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
        f.write(source)
    print("Patched successfully")
else:
    print("Could not find insertion point")
