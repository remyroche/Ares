import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    content = f.read()

content = re.sub(r'out\["mask_opt_max_symbols"\] = 200', 'out["mask_opt_max_symbols"] = 100', content)
content = re.sub(r'out\["mask_opt_lookback_years"\] = 3\.0', 'out["mask_opt_lookback_years"] = 3.0', content)

with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
    f.write(content)
