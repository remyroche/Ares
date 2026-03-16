with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    code = f.read()

code = code.replace(
'''def display_arity_for_key(canonical_key: str) -> int:
    composite_parts = split_composite_key(canonical_key)
    if composite_parts is not None:
        return max(display_arity_for_key(part) for part in composite_parts)
    return sum(slot.strip("()") != "*" for slot in canonical_key.split("|"))''',
'''def display_arity_for_key(canonical_key: str) -> int:
    composite_parts = split_composite_key(canonical_key)
    if composite_parts is not None:
        return max(display_arity_for_key(part) for part in composite_parts)

    total = 0
    for slot in canonical_key.split("|"):
        slot_value = slot.strip("()")
        if slot_value == "*":
            continue
        total += sum(1 for cond_str in slot_value.split("&") if "==" in cond_str)
    return total'''
)

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(code)
