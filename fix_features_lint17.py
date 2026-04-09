with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

content = content.replace('ff.numba_rolling_mean(ret1h, 24)', 'ff.apply_to_frame(feats["ret1h"], ff._numba_rolling_mean_nan_safe, 24).astype(np.float32)')
content = content.replace('ff.numba_rolling_mean(v, 24)', 'ff.apply_to_frame(v, ff._numba_rolling_mean_nan_safe, 24).astype(np.float32)')

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
