import pandas as pd
import numpy as np
from typing import Dict

def select_top_diverse_rules(
    registry: pd.DataFrame,
    mask_map: Dict[str, np.ndarray],
    top_n: int = 15,
    max_overlap: float = 0.4,
    max_side_in_top10: int = 6,
) -> pd.DataFrame:
    """
    Select top `top_n` diverse rules:
    - Sort by composite_score
    - Ensure top 10 has at most `max_side_in_top10` of the same side (long/short)
      IF there are enough valid rules of the other side to fill the quota.
    - Ensure jaccard similarity between any two selected rules is <= max_overlap
    """
    if registry.empty:
        return registry

    sorted_reg = registry.sort_values("composite_score", ascending=False)

    selected_idx = []
    selected_sides = {"long": 0, "short": 0}

    for idx, row in sorted_reg.iterrows():
        if len(selected_idx) >= top_n:
            break

        key = row["canonical_key"]
        side = row.get("side", "unknown")
        mask = mask_map.get(key)
        if mask is None:
            continue

        # Check side constraint only for the first 10
        if len(selected_idx) < 10 and side in selected_sides:
            if selected_sides[side] >= max_side_in_top10:
                other_side = "short" if side == "long" else "long"
                slots_to_fill = 10 - len(selected_idx)
                valid_other_side = 0

                # Check remaining items
                remaining_indices = sorted_reg.index.drop(selected_idx)
                # But only check those after the current idx in the sorted series
                curr_pos = sorted_reg.index.get_loc(idx)
                remaining_indices = sorted_reg.index[curr_pos + 1:]

                for rem_idx in remaining_indices:
                    rem_row = sorted_reg.loc[rem_idx]
                    if rem_row.get("side", "unknown") == other_side:
                        rem_mask = mask_map.get(rem_row["canonical_key"])
                        if rem_mask is not None:
                            too_similar = False
                            for s_idx in selected_idx:
                                s_mask = mask_map.get(sorted_reg.loc[s_idx, "canonical_key"])
                                intersection = float(np.sum(rem_mask & s_mask))
                                union = float(np.sum(rem_mask | s_mask))
                                jaccard = intersection / union if union > 0 else 0.0
                                if jaccard > max_overlap:
                                    too_similar = True
                                    break

                            if not too_similar:
                                valid_other_side += 1
                                if valid_other_side >= slots_to_fill:
                                    break

                # If we have enough valid rules of the other side to fill the 10 spots,
                # skip this one. Otherwise, allow the side count to exceed max_side_in_top10.
                if valid_other_side >= slots_to_fill:
                    continue

        # Check overlap constraint
        too_similar = False
        for s_idx in selected_idx:
            s_key = sorted_reg.loc[s_idx, "canonical_key"]
            s_mask = mask_map.get(s_key)
            if s_mask is None:
                continue

            intersection = float(np.sum(mask & s_mask))
            union = float(np.sum(mask | s_mask))
            jaccard = intersection / union if union > 0 else 0.0

            if jaccard > max_overlap:
                too_similar = True
                break

        if not too_similar:
            selected_idx.append(idx)
            if len(selected_idx) <= 10 and side in selected_sides:
                selected_sides[side] += 1

    if len(selected_idx) < min(top_n, len(registry)) and max_overlap < 0.8:
        return select_top_diverse_rules(
            registry, mask_map, top_n, max_overlap + 0.1, max_side_in_top10
        )

    return sorted_reg.loc[selected_idx]

# Test 1: We have 50 longs, 2 shorts. Quota is 10 rules, max 6 longs.
# We should get 8 longs, 2 shorts (10 total rules).
print("Test 1:")
data = []
mask_map = {}
for i in range(50):
    key = f"rule_{i}"
    data.append({"canonical_key": key, "composite_score": 100 - i, "side": "long"})
    mask = np.zeros(200, dtype=bool)
    mask[i*4:i*4+4] = True
    mask_map[key] = mask

for i in range(2):
    key = f"rule_s_{i}"
    data.append({"canonical_key": key, "composite_score": 5 - i, "side": "short"})
    mask = np.zeros(200, dtype=bool)
    mask[150+i*4:150+i*4+4] = True
    mask_map[key] = mask

registry = pd.DataFrame(data)
top = select_top_diverse_rules(registry, mask_map, top_n=10, max_overlap=0.4, max_side_in_top10=6)
print(f"Got {len(top)} rules")
print(top["side"].value_counts())

# Test 2: We have 50 longs, 50 shorts.
# We should get 6 longs, 4 shorts (or similar, respecting the 6 limit).
print("\nTest 2:")
data2 = []
mask_map2 = {}
for i in range(50):
    key = f"rule_{i}"
    data2.append({"canonical_key": key, "composite_score": 100 - i, "side": "long"})
    mask = np.zeros(500, dtype=bool)
    mask[i*4:i*4+4] = True
    mask_map2[key] = mask

for i in range(50):
    key = f"rule_s_{i}"
    data2.append({"canonical_key": key, "composite_score": 80 - i, "side": "short"})
    mask = np.zeros(500, dtype=bool)
    mask[250+i*4:250+i*4+4] = True
    mask_map2[key] = mask

registry2 = pd.DataFrame(data2)
top2 = select_top_diverse_rules(registry2, mask_map2, top_n=10, max_overlap=0.4, max_side_in_top10=6)
print(f"Got {len(top2)} rules")
print(top2["side"].value_counts())

# Test 3: We have 50 longs, 4 shorts.
# We should get 6 longs, 4 shorts.
print("\nTest 3:")
data3 = []
mask_map3 = {}
for i in range(50):
    key = f"rule_{i}"
    data3.append({"canonical_key": key, "composite_score": 100 - i, "side": "long"})
    mask = np.zeros(500, dtype=bool)
    mask[i*4:i*4+4] = True
    mask_map3[key] = mask

for i in range(4):
    key = f"rule_s_{i}"
    data3.append({"canonical_key": key, "composite_score": 80 - i, "side": "short"})
    mask = np.zeros(500, dtype=bool)
    mask[250+i*4:250+i*4+4] = True
    mask_map3[key] = mask

registry3 = pd.DataFrame(data3)
top3 = select_top_diverse_rules(registry3, mask_map3, top_n=10, max_overlap=0.4, max_side_in_top10=6)
print(f"Got {len(top3)} rules")
print(top3["side"].value_counts())

# Test 4: We want top 15, max 6 longs in top 10.
print("\nTest 4:")
top4 = select_top_diverse_rules(registry2, mask_map2, top_n=15, max_overlap=0.4, max_side_in_top10=6)
print(f"Got {len(top4)} rules")
print(top4["side"].value_counts())
