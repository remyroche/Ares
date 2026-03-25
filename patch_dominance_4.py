import re

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

content = re.sub(
    r"    - not dominated by simpler parent unless uplift is material\n",
    "",
    content
)

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(content)
