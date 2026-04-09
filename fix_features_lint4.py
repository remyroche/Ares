with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

# Replace ret1h_std_96 with ff.numba_rolling_std(feats["ret1h"], 96) in the problematic locations (lines 948-967)
lines = content.split("\n")
for i in range(945, 975):
    lines[i] = lines[i].replace("ret1h_std_96", 'ff.numba_rolling_std(feats["ret1h"], 96)')

with open("extreme_price_movements/features.py", "w") as f:
    f.write("\n".join(lines))
