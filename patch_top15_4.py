import sys

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

import re
matches = list(re.finditer(r"top_final = select_top_diverse_rules\([\s\S]*?\)", content))
for m in matches:
    print(f"Match found at position {m.start()}:")
    print(content[m.start()-100:m.end()+200])
