with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

content = content.replace('feats["seasonality_strength"] = (feats["ret1h"] - hourly_avg_ret).abs().astype(np.float32)', 'feats["seasonality_strength"] = (ret1h - hourly_avg_ret).abs().astype(np.float32)')

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
