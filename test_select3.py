import pandas as pd
from extreme_price_movements.lgbm_based_mask_generation import select_top_diverse_rules
import numpy as np

# Create mock registry
data = []
mask_map = {}
for i in range(50):
    key = f"rule_{i}"
    side = "long"
    data.append({
        "canonical_key": key,
        "composite_score": 100 - i,
        "side": side,
        "hurdle_excess": 0.05
    })

    # Create masks with some overlap
    mask = np.zeros(100, dtype=bool)
    if i < 10:
        mask[0:10] = True # First 10 overlap
    else:
        mask[i:i+5] = True
    mask_map[key] = mask

for i in range(50):
    key = f"rule_s_{i}"
    side = "short"
    data.append({
        "canonical_key": key,
        "composite_score": 50 - i,
        "side": side,
        "hurdle_excess": 0.05
    })

    # Create masks with some overlap
    mask = np.zeros(100, dtype=bool)
    if i < 10:
        mask[50:60] = True # First 10 overlap
    else:
        mask[i+50:i+55] = True
    mask_map[key] = mask

registry = pd.DataFrame(data)

# Sort by hurdle_excess then composite_score
top = select_top_diverse_rules(registry, mask_map, top_n=10, max_overlap=0.4, max_side_in_top10=6)
print(f"Got {len(top)} rules")
print(top["side"].value_counts())
