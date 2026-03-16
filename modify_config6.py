import re

with open('extreme_price_movements/config.py', 'r') as f:
    content = f.read()

new_meta = {
    # Trend
    "ema20_gt_ema50": {"family": "trend", "type": "binary"},
    "ema50_gt_ema200": {"family": "trend", "type": "binary"},
    "price_lt_ema200": {"family": "trend", "type": "binary"},
    "ema50_slope": {"family": "trend", "type": "continuous"},
    "trend_strength_percentile": {"family": "trend", "type": "continuous"},
    "ema20_slope_5h": {"family": "trend", "type": "continuous"},
    "ema_slope_norm": {"family": "trend", "type": "continuous"},
    "trend_persistence": {"family": "trend", "type": "continuous"},
    "trend_ratio": {"family": "trend", "type": "continuous"},
    "trend_acceleration": {"family": "trend", "type": "continuous"},
    "return_autocorr_48": {"family": "trend", "type": "continuous"},
    "variance_ratio_10_48": {"family": "trend", "type": "continuous"},

    # Volatility Level
    "rolling_std_4h": {"family": "volatility_level", "type": "continuous"},
    "realized_volatility_24h": {"family": "volatility_level", "type": "continuous"},
    "true_range_percentile": {"family": "volatility_level", "type": "continuous"},
    "atr_percentile": {"family": "volatility_level", "type": "continuous"},
    "rolling_range_20": {"family": "volatility_level", "type": "continuous"},
    "volatility_of_volatility_48": {"family": "volatility_level", "type": "continuous"},
    "volatility_autocorr_48": {"family": "volatility_level", "type": "continuous"},

    # Volatility Change
    "atr_change_rate": {"family": "volatility_change", "type": "continuous"},
    "compression_ratio": {"family": "volatility_change", "type": "continuous"},
    "range_expansion_ratio": {"family": "volatility_change", "type": "continuous"},
    "compression_score": {"family": "volatility_change", "type": "continuous"},
    "atr_compression_ratio": {"family": "volatility_change", "type": "continuous"},
    "volatility_ratio_short_long": {"family": "volatility_change", "type": "continuous"},
    "bollinger_band_width": {"family": "volatility_change", "type": "continuous"},

    # Path Structure
    "efficiency_ratio_20": {"family": "path_structure", "type": "continuous"},
    "choppiness_index_20": {"family": "path_structure", "type": "continuous"},
    "direction_entropy_20": {"family": "path_structure", "type": "continuous"},

    # Liquidity
    "volume_percentile": {"family": "liquidity", "type": "continuous"},
    "volume_zscore_48h": {"family": "liquidity", "type": "continuous"},
    "volume_trend_48": {"family": "liquidity", "type": "continuous"},
    "volume_autocorr_48": {"family": "liquidity", "type": "continuous"},
}

def format_dict(d):
    lines = ["RIDGE_FEATURE_META = {"]
    for k, v in d.items():
        lines.append(f'    "{k}": {{"family": "{v["family"]}", "type": "{v["type"]}"}},')
    lines.append("}")
    return '\n'.join(lines)

new_meta_str = format_dict(new_meta)

def replace_dict(s, start_str):
    start_idx = s.find(start_str)
    if start_idx == -1: return s

    dict_start = s.find("{", start_idx)
    stack = []
    end_idx = -1
    for i in range(dict_start, len(s)):
        if s[i] == '{':
            stack.append('{')
        elif s[i] == '}':
            stack.pop()
            if not stack:
                end_idx = i + 1
                break

    return s[:start_idx] + new_meta_str + s[end_idx:]

content = replace_dict(content, "RIDGE_FEATURE_META = {")

# Add prior_range, prior_volatility to CONTINUOUS_TRIGGER_COLS
trig_features = ["prior_range", "prior_volatility"]
trig_block_match = re.search(r'CONTINUOUS_TRIGGER_COLS = \[(.*?)\]', content, re.DOTALL)
if trig_block_match:
    trig_block = trig_block_match.group(1)
    existing_trigs = [x.strip().strip('"').strip("'") for x in trig_block.split(',') if x.strip()]

    for f in trig_features:
        if f not in existing_trigs:
            existing_trigs.append(f)

    new_trig_block = 'CONTINUOUS_TRIGGER_COLS = [\n' + ',\n'.join([f'    "{f}"' for f in existing_trigs]) + ',\n]'
    content = content[:trig_block_match.start()] + new_trig_block + content[trig_block_match.end():]

with open('extreme_price_movements/config.py', 'w') as f:
    f.write(content)
