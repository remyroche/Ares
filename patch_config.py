import sys

def patch_file():
    filepath = 'extreme_price_movements/config.py'
    with open(filepath, 'r') as f:
        content = f.read()

    new_keys = """    # Longer-timeframe regime context
    "vol_regime_z", "regime_stability_24h",
    "bars_since_ema20_ema50_cross_log_norm",
    "bars_in_high_vol_state_log_norm",
    "bars_outside_ema20_atr_band_log_norm",
    "up_down_semivol_ratio_tanh",
    "up_down_return_mass_ratio_tanh",
    "tail_asymmetry_q90_q10_atr_norm","""

    content = content.replace("""    # Longer-timeframe regime context
    "vol_regime_z", "regime_stability_24h",""", new_keys)

    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    patch_file()
