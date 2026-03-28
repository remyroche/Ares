import pandas as pd
from extreme_price_movements.lgbm_based_mask_generation import select_top_diverse_rules
import numpy as np

# Create mock registry
data = []
mask_map = {}
for i in range(50):
    key = f"rule_{i}"
    side = "long" if i % 2 == 0 else "short"
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

registry = pd.DataFrame(data)

# Sort by hurdle_excess then composite_score
top = select_top_diverse_rules(registry, mask_map, top_n=15, max_overlap=0.4, max_side_in_top10=6)
print(f"Got {len(top)} rules")
print(top["side"].value_counts())
