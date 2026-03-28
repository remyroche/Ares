import sys
import re

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# Replace the select_top_diverse_rules function
pattern = r"def select_top_diverse_rules\((.*?)\) -> pd\.DataFrame:.*?return sorted_reg\.loc\[selected_idx\]"
# We need to use re.DOTALL
match = re.search(pattern, content, re.DOTALL)
if not match:
    print("Could not find select_top_diverse_rules")
    sys.exit(1)

new_func = """def select_top_diverse_rules(
    registry: pd.DataFrame,
    mask_map: Dict[str, np.ndarray],
    top_n: int = 15,
    max_overlap: float = 0.4,
    max_side_in_top: int = 9,
) -> pd.DataFrame:
    \"\"\"
    Select top `top_n` diverse rules:
    - Sort by composite_score
    - Ensure top `top_n` has at most `max_side_in_top` of the same side (long/short)
      IF there are enough valid rules of the other side to fill the quota.
    - Ensure jaccard similarity between any two selected rules is <= max_overlap
    \"\"\"
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

        # Check side constraint for the selected rules
        if len(selected_idx) < top_n and side in selected_sides:
            if selected_sides[side] >= max_side_in_top:
                other_side = "short" if side == "long" else "long"
                slots_to_fill = top_n - len(selected_idx)
                valid_other_side = 0

                # Check remaining items
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

                # If we have enough valid rules of the other side to fill the top_n spots,
                # skip this one. Otherwise, allow the side count to exceed max_side_in_top.
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
            if len(selected_idx) <= top_n and side in selected_sides:
                selected_sides[side] += 1

    if len(selected_idx) < min(top_n, len(registry)) and max_overlap < 0.8:
        return select_top_diverse_rules(
            registry, mask_map, top_n, max_overlap + 0.1, max_side_in_top
        )

    return sorted_reg.loc[selected_idx]"""

new_content = content[:match.start()] + new_func + content[match.end():]

# We need to change where it's called
# find top_final = select_top_diverse_rules(
#             combined_global_registry, combined_mask_map, top_n=15
#         )
# and change `max_side_in_top10` to `max_side_in_top` if it was passed, but it's not.
with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(new_content)

print("Patch applied successfully")
