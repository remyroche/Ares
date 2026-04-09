with open("extreme_price_movements/features.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "Features: {len(feats)} features before CausalTransform. Applying transforms..." in line:
        lines[i] = '    tprint(f"Features: {len(feats)} features before CausalTransform. Applying transforms...")\n'

with open("extreme_price_movements/features.py", "w") as f:
    f.writelines(lines)
