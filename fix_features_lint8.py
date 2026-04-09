with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

# Revert my change from earlier where I made an assignment `ret1h_std_96 = ff.numba_rolling_std...` inside `_compute_features_impl` instead of `compute_regime_features`. Let's just fix the usage inside `compute_regime_features` to use `rv_24` instead of `ret1h_std_96`. Because `ret1h_std_96` is simply the rolling std of 96 bars (24 hours in 15min), which is `rv_24` in hourly context. Wait, no.

lines = content.split('\n')
for i, line in enumerate(lines):
    if "hour_vol = _roll_std" in line:
        lines[i] = '    hour_vol = ff.numba_rolling_std(ret1h, 4)'
    elif "feats[\"hour_vol_ratio\"] =" in line and "ret1h_std_96" in line:
        lines[i] = '    feats["hour_vol_ratio"] = (hour_vol / (rv_24 + 1e-12)).astype(np.float32)'
    elif "jump_t =" in line and "ret1h_std_96" in line:
        lines[i] = '    jump_t = (ret1h.abs() > 3 * rv_24).astype(np.float32)'
    elif "short_vol = _roll_std" in line:
        lines[i] = '    short_vol = ff.numba_rolling_std(ret1h, 12)'
    elif "long_vol = ret1h_std_96" in line:
        lines[i] = '    long_vol = rv_24'
    elif "feats[\"trend_strength\"] =" in line and "ret1h_std_96" in line:
        lines[i] = '    feats["trend_strength"] = ((ema_fast - ema_slow).abs() / (rv_24 + 1e-12)).astype(np.float32)'
    elif "hourly_avg_ret =" in line and "_roll_mean" in line:
        lines[i] = '    hourly_avg_ret = ff.numba_rolling_mean(ret1h, 24)'
    elif "rolling_volume = _roll_mean" in line:
        lines[i] = '    rolling_volume = ff.numba_rolling_mean(v, 24)'

with open("extreme_price_movements/features.py", "w") as f:
    f.write('\n'.join(lines))
