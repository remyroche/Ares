with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

content = content.replace('ff.numba_rolling_std(feats["ret1h"], 96) + 1e-12', 'ret1h_std_96 + 1e-12')
content = content.replace('ff.numba_rolling_std(feats["ret1h"], 4)', '_roll_std("ret1h", feats["ret1h"], 4)')
content = content.replace('ff.numba_rolling_std(feats["ret1h"], 12)', '_roll_std("ret1h", feats["ret1h"], 12)')
content = content.replace('ff.numba_rolling_mean(feats["ret1h"], 24)', '_roll_mean("ret1h", feats["ret1h"], 24)')
content = content.replace('ff.numba_rolling_mean(v, 24)', '_roll_mean("volume", v, 24)')
content = content.replace('ff.numba_rolling_std(feats["ret1h"], 96)', 'ret1h_std_96')

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
