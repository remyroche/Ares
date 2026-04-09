with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

content = content.replace('ff.numba_rolling_std(ret1h, 96) = _roll_std("ret1h", feats["ret1h"], 96)', 'ret1h_std_96_temp = _roll_std("ret1h", feats["ret1h"], 96)')
content = content.replace('feats["realized_volatility_24h"] = ff.numba_rolling_std(ret1h, 96)  # 24h = 96 * 15m', 'feats["realized_volatility_24h"] = ret1h_std_96_temp  # 24h = 96 * 15m')

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
