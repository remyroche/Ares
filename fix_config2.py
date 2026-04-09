with open("extreme_price_movements/config.py", "r") as f:
    content = f.read()

old_list = """TRAINING_RESIDUALIZATION_FEATURE_KEYS = [
    "overext_surprise",
    "blowoff_risk_surprise",
    "exh_qual_surprise",
    "dist_vwap_resid",
    "dist_ema_fast_resid",
    "trend_pct_resid",
]"""

new_list = """TRAINING_RESIDUALIZATION_FEATURE_KEYS = [
    "ema50_ema200_spread_continuous",
    "atr_change_rate_ts_continuous",
    "bars_in_high_vol_state_log_norm",
    "volatility_of_volatility_48",
    "trend_strength_percentile",
    "volatility_autocorr_48",
]"""

content = content.replace(old_list, new_list)

with open("extreme_price_movements/config.py", "w") as f:
    f.write(content)
