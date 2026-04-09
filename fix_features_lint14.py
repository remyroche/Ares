with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

content = content.replace('_roll_std("ret1h", feats["ret1h"], 4)', 'ff.numba_rolling_std(ret1h.to_numpy(), 4)')
content = content.replace('_roll_std("ret1h", feats["ret1h"], 12)', 'ff.numba_rolling_std(ret1h.to_numpy(), 12)')

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
