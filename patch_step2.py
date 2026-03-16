with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    code = f.read()

code = code.replace(
'''def extract_feature_names_from_key(canonical_key: str) -> List[str]:
    names: List[str] = []
    for part in iter_primitive_keys(canonical_key):
        for slot in part.split("|"):
            slot_value = slot.strip("()")
            if slot_value == "*" or "==" not in slot_value:
                continue
            names.append(slot_value.split("==")[0])
    return sorted(set(names))''',
'''def extract_feature_names_from_key(canonical_key: str) -> List[str]:
    names: List[str] = []
    for part in iter_primitive_keys(canonical_key):
        for slot in part.split("|"):
            slot_value = slot.strip("()")
            if slot_value == "*":
                continue
            for cond_str in slot_value.split("&"):
                if "==" not in cond_str:
                    continue
                names.append(cond_str.split("==")[0])
    return sorted(set(names))'''
)

# Also apply P2-9 (iter_primitive_keys) to ensure it uses the safe parsing strategy if it didn't already
# The current one looks like:
# def iter_primitive_keys(canonical_key: str) -> List[str]:
#    composite_parts = split_composite_key(canonical_key)
#    if composite_parts is None:
#        return [canonical_key]
#    out: List[str] = []
#    for part in composite_parts:
#        out.extend(iter_primitive_keys(part))
#    return out

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(code)
