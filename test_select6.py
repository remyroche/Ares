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
            # Check if adding this would exceed max_side_in_top10
            # BUT we also need to ensure we CAN fill the rest of the 10 spots with the OTHER side.
            # E.g. if we have 6 longs and 0 shorts, we can only add shorts IF there are at least 4 valid shorts available.
            # If there aren't 4 shorts available, we shouldn't arbitrarily limit longs to 6, otherwise we won't get 10 rules.
            # For simplicity, we can do a softer constraint: if we hit the limit, see if there are enough of the other side left.
            # Let's count available of other side.
            if selected_sides[side] >= max_side_in_top10:
                # How many slots left to reach 10?
                slots_left = 10 - len(selected_idx)

                # Are there enough valid rules of the OTHER side to fill these slots?
                other_side = "short" if side == "long" else "long"
                # Find remaining valid rules of other_side
                available_other = 0
                for rem_idx, rem_row in sorted_reg.loc[~sorted_reg.index.isin(selected_idx)].iterrows():
                    if rem_row.get("side", "unknown") == other_side:
                        rem_mask = mask_map.get(rem_row["canonical_key"])
                        if rem_mask is not None:
                            # Fast check overlap with already selected
                            too_similar = False
                            for s_idx in selected_idx:
                                s_key = sorted_reg.loc[s_idx, "canonical_key"]
                                s_mask = mask_map.get(s_key)
                                intersection = float(np.sum(rem_mask & s_mask))
                                union = float(np.sum(rem_mask | s_mask))
                                jaccard = intersection / union if union > 0 else 0.0
                                if jaccard > max_overlap:
                                    too_similar = True
                                    break
                            if not too_similar:
                                available_other += 1
                                if available_other >= slots_left:
                                    break

                if available_other >= slots_left:
                    continue # Skip this one, we have enough of the other side to fill the quota
                # Else: we can't fill the quota with the other side, so we MUST allow this one to exceed the max_side limit

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
