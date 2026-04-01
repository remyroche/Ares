import sys

def patch_file():
    filepath = 'extreme_price_movements/config.py'
    with open(filepath, 'r') as f:
        content = f.read()

    to_remove = """    # Longer-timeframe regime context
    "vol_regime_z",
    "regime_stability_24h",
    "bars_since_ema20_ema50_cross_log_norm",
    "bars_in_high_vol_state_log_norm",
    "bars_outside_ema20_atr_band_log_norm",
    "up_down_semivol_ratio_tanh",
    "up_down_return_mass_ratio_tanh",
    "tail_asymmetry_q90_q10_atr_norm",
"""
    replace_with = """    # Longer-timeframe regime context
    "vol_regime_z",
    "regime_stability_24h",
"""

    # Simple replace
    new_content = content.replace(to_remove, replace_with)

    # Try alternate replacement for RIDGE_FEATURE_META
    lines = new_content.splitlines()
    out_lines = []
    in_ridge = False
    for line in lines:
        if "RIDGE_FEATURE_META = {" in line:
            in_ridge = True
            out_lines.append(line)
        elif in_ridge and line.strip() == "}":
            in_ridge = False
            out_lines.append('    "bars_since_ema20_ema50_cross_log_norm": {"family": "trend", "type": "continuous"},')
            out_lines.append('    "bars_in_high_vol_state_log_norm": {"family": "volatility", "type": "continuous"},')
            out_lines.append('    "bars_outside_ema20_atr_band_log_norm": {"family": "volatility", "type": "continuous"},')
            out_lines.append('    "up_down_semivol_ratio_tanh": {"family": "path_structure", "type": "continuous"},')
            out_lines.append('    "up_down_return_mass_ratio_tanh": {"family": "path_structure", "type": "continuous"},')
            out_lines.append('    "tail_asymmetry_q90_q10_atr_norm": {"family": "path_structure", "type": "continuous"},')
            out_lines.append(line)
        else:
            out_lines.append(line)

    with open(filepath, 'w') as f:
        f.write("\n".join(out_lines) + "\n")

if __name__ == "__main__":
    patch_file()
