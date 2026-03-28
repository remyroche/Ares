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
                # Count valid remaining rules of the *other* side
                other_side = "short" if side == "long" else "long"
                slots_to_fill = 10 - len(selected_idx)
                valid_other_side = 0

                # Look ahead in the remaining registry
                remaining_indices = sorted_reg.index.difference(selected_idx)
                for rem_idx in remaining_indices[remaining_indices > idx]:
                    rem_row = sorted_reg.loc[rem_idx]
                    if rem_row.get("side", "unknown") == other_side:
                        rem_mask = mask_map.get(rem_row["canonical_key"])
                        if rem_mask is not None:
                            # Check if it overlaps with already selected rules
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

                # If we have enough valid rules of the other side to reach 10, skip this one
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

    # If we couldn't find enough rules, we could try relaxing the overlap constraint slightly
    if len(selected_idx) < min(top_n, len(registry)) and max_overlap < 0.8:
        # Recursive call with relaxed overlap
        return select_top_diverse_rules(
            registry, mask_map, top_n, max_overlap + 0.1, max_side_in_top10
        )

    return sorted_reg.loc[selected_idx]

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
    mask = np.zeros(200, dtype=bool)
    mask[i*4:i*4+4] = True
    mask_map[key] = mask

for i in range(2):
    key = f"rule_s_{i}"
    side = "short"
    data.append({
        "canonical_key": key,
        "composite_score": 5 - i,
        "side": side,
        "hurdle_excess": 0.05
    })

    # Create masks with some overlap
    mask = np.zeros(200, dtype=bool)
    mask[150+i*4:150+i*4+4] = True
    mask_map[key] = mask

registry = pd.DataFrame(data)

# Sort by hurdle_excess then composite_score
top = select_top_diverse_rules(registry, mask_map, top_n=10, max_overlap=0.4, max_side_in_top10=6)
print(f"Got {len(top)} rules")
print(top["side"].value_counts())
