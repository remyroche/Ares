with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

# Replace the rv_48h logic to use _safe_log_ratio instead of _safe_div
old_rv48 = """    if scale_rv_long is not None and "rv_48h" in feats:
        feats["rv_48h"] = _safe_div(feats["rv_48h"], scale_rv_long).astype(np.float32)"""

new_rv48 = """    if scale_rv_long is not None and "rv_48h" in feats:
        feats["rv_48h"] = _safe_log_ratio(feats["rv_48h"], scale_rv_long).astype(np.float32)"""

content = content.replace(old_rv48, new_rv48)

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
