with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# Instead of using select_top_diverse_rules, we will create a new function `select_final_regimes`
# that implements the logic: "greedy, order-dependent selection. Sort candidates by base_regime_score descending,
# recompute overlap, compute selection_score, accept/reject, keep top 10".

# I'll define `dice_overlap(mask_a, mask_b)`.
new_funcs = """
def dice_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = float(np.sum(mask_a & mask_b))
    support_a = float(np.sum(mask_a))
    support_b = float(np.sum(mask_b))
    if support_a + support_b == 0:
        return 0.0
    return (2.0 * intersection) / (support_a + support_b)

def select_final_regimes(
    registry: pd.DataFrame,
    mask_map: Dict[str, np.ndarray],
    top_n: int = 10,
) -> pd.DataFrame:
    if registry.empty:
        return registry

    # Filter out structurally unsound
    if "is_structurally_sound" in registry.columns:
        valid_reg = registry[registry["is_structurally_sound"]].copy()
    else:
        valid_reg = registry.copy()

    if valid_reg.empty:
        return valid_reg

    # Sort candidates by base_regime_score descending
    # Assume base_regime_score is stored in 'regime_score'
    sorted_reg = valid_reg.sort_values("regime_score", ascending=False).reset_index(drop=True)

    selected_rows = []
    accepted_masks = []

    # Pre-union mask
    if len(sorted_reg) > 0:
        shape = mask_map[sorted_reg.iloc[0]["canonical_key"]].shape
        accepted_union_mask = np.zeros(shape, dtype=bool)
    else:
        return sorted_reg

    for idx, row in sorted_reg.iterrows():
        if len(selected_rows) >= top_n:
            break

        key = str(row["canonical_key"])
        mask = mask_map.get(key)
        if mask is None:
            continue

        base_score = float(row.get("regime_score", 0.0))
        worst_penalty = float(row.get("worst_penalty", 1.0))

        # Calculate overlaps
        pairwise_raw_overlap = 0.0
        if accepted_masks:
            eligible_pairwise_overlaps = [dice_overlap(mask, m) for m in accepted_masks]
            pairwise_raw_overlap = max(eligible_pairwise_overlaps, default=0.0)

        pairwise_overlap_penalty = pairwise_raw_overlap if pairwise_raw_overlap >= 0.30 else 0.0

        if accepted_masks:
            union_raw_overlap = dice_overlap(mask, accepted_union_mask)
        else:
            union_raw_overlap = 0.0

        union_overlap_penalty = union_raw_overlap if union_raw_overlap >= 0.40 else 0.0

        # Hard reject conditions
        if pairwise_raw_overlap >= 0.50:
            continue
        if union_raw_overlap >= 0.75:
            continue

        overlap_penalty = max(pairwise_overlap_penalty, 0.70 * union_overlap_penalty)

        selection_score = (
            base_score
            - 0.2 * (overlap_penalty ** 2)
            - 0.1 * (worst_penalty ** 2)
        )

        row_dict = row.to_dict()
        row_dict["selection_score"] = selection_score
        selected_rows.append(row_dict)

        accepted_masks.append(mask)
        accepted_union_mask |= mask

    if not selected_rows:
        return pd.DataFrame()

    final_df = pd.DataFrame(selected_rows)
    return final_df.sort_values("selection_score", ascending=False).reset_index(drop=True)
"""

# Insert `new_funcs` before `select_top_diverse_rules`
if "def select_top_diverse_rules" in source:
    source = source.replace("def select_top_diverse_rules", new_funcs + "\n\ndef select_top_diverse_rules")
    with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
        f.write(source)
    print("Injected selection logic")
else:
    print("Could not find select_top_diverse_rules")
