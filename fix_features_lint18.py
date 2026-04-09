with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

content = content.replace('_roll_std("ret1h", feats["ret1h"], 4)', 'ff.apply_to_frame(ret1h, ff._numba_rolling_std_nan_safe, 4).astype(np.float32)')
content = content.replace('_roll_std("ret1h", feats["ret1h"], 12)', 'ff.apply_to_frame(ret1h, ff._numba_rolling_std_nan_safe, 12).astype(np.float32)')

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
