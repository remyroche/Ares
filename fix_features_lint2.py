with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

# Ruff complained about `_roll_std`, `ret1h_std_96` because they are defined inside `_compute_features_impl` but used inside `compute_regime_features` (lines 953, etc.)
# Let's fix those by replacing them with `ff.numba_rolling_std` since `_roll_std` uses it internally anyway.

replacements = {
    '_roll_std("ret1h", feats["ret1h"], 4)': 'ff.numba_rolling_std(feats["ret1h"], 4)',
    '_roll_std("ret1h", feats["ret1h"], 12)': 'ff.numba_rolling_std(feats["ret1h"], 12)',
    '_roll_mean("ret1h", feats["ret1h"], 24)': 'ff.numba_rolling_mean(feats["ret1h"], 24)',
    '_roll_mean("volume", v, 24)': 'ff.numba_rolling_mean(v, 24)',
    'ret1h_std_96': 'ff.numba_rolling_std(feats["ret1h"], 96)'
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
