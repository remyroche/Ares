with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

# Fix syntax error
content = content.replace('ff.numba_rolling_std(feats["ret1h"], 96) = _roll_std("ret1h", feats["ret1h"], 96)', 'ret1h_std_96 = ff.numba_rolling_std(feats["ret1h"], 96)')
content = content.replace('feats["realized_volatility_24h"] = ff.numba_rolling_std(feats["ret1h"], 96)', 'feats["realized_volatility_24h"] = ret1h_std_96')
content = content.replace('ff.numba_rolling_std(feats["ret1h"], 96) + 1e-12', 'ret1h_std_96 + 1e-12')
content = content.replace('3 * ff.numba_rolling_std(feats["ret1h"], 96)', '3 * ret1h_std_96')
content = content.replace('long_vol = ff.numba_rolling_std(feats["ret1h"], 96)', 'long_vol = ret1h_std_96')

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
