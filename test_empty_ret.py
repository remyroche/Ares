with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

old_empty = """        if not np.any(mask):
            return np.nan, 0.0"""
new_empty = """        if not np.any(mask):
            return np.nan, 0.0, np.full(len(X), np.nan)"""

if old_empty in source:
    source = source.replace(old_empty, new_empty)

old_empty2 = """        if not ridge_feats:
            return np.nan, 0.0"""
new_empty2 = """        if not ridge_feats:
            return np.nan, 0.0, np.full(len(X), np.nan)"""

if old_empty2 in source:
    source = source.replace(old_empty2, new_empty2)

with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
    f.write(source)

print("Patched empty returns")
