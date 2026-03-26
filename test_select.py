import pandas as pd
import numpy as np

def select_top_diverse_rules(registry, mask_map, top_n=15, max_overlap=0.4, max_side_in_top10=6):
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
    selected_sides = {'long': 0, 'short': 0}

    for idx, row in sorted_reg.iterrows():
        if len(selected_idx) >= top_n:
            break

        key = row['canonical_key']
        side = row['side']
        mask = mask_map.get(key)
        if mask is None:
            continue

        # Check side constraint only for the first 10
        if len(selected_idx) < 10:
            if selected_sides.get(side, 0) >= max_side_in_top10:
                continue

        # Check overlap constraint
        too_similar = False
        for s_idx in selected_idx:
            s_key = sorted_reg.loc[s_idx, 'canonical_key']
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
            if len(selected_idx) < 10:
                selected_sides[side] = selected_sides.get(side, 0) + 1

    # If we couldn't find enough rules, we could try relaxing the overlap constraint slightly
    if len(selected_idx) < min(top_n, len(registry)) and max_overlap < 0.8:
        # Recursive call with relaxed overlap
        return select_top_diverse_rules(registry, mask_map, top_n, max_overlap + 0.1, max_side_in_top10)

    return sorted_reg.loc[selected_idx]

# Let's write a mock test for it.
df = pd.DataFrame({
    'canonical_key': ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13'],
    'side': ['long', 'long', 'long', 'long', 'long', 'long', 'long', 'short', 'short', 'short', 'short', 'short', 'short'],
    'composite_score': [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88]
})

mask_map = {
    'r1': np.array([1, 0, 0, 0]),
    'r2': np.array([0, 1, 0, 0]),
    'r3': np.array([0, 0, 1, 0]),
    'r4': np.array([0, 0, 0, 1]),
    'r5': np.array([1, 1, 0, 0]),
    'r6': np.array([1, 0, 1, 0]),
    'r7': np.array([1, 0, 0, 1]),
    'r8': np.array([0, 1, 1, 0]),
    'r9': np.array([0, 1, 0, 1]),
    'r10': np.array([0, 0, 1, 1]),
    'r11': np.array([1, 1, 1, 0]),
    'r12': np.array([1, 1, 0, 1]),
    'r13': np.array([1, 0, 1, 1]),
}

res = select_top_diverse_rules(df, mask_map, top_n=10, max_overlap=0.1, max_side_in_top10=6)
print(res)
print(len(res))
print(res['side'].value_counts())
