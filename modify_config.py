import re

with open('extreme_price_movements/config.py', 'r') as f:
    content = f.read()

# 1. Features to move to CONTINUOUS_LOCATION_COLS and remove from RIDGE_FEATURE_META
loc_features = [
    "dist_ema20_atr", "dist_ema50_atr", "dist_ema200_atr", "dist_vwap_atr",
    "dist_weekly_vwap", "dist_prior_day_high", "dist_prior_day_low",
    "dist_rolling_7d_high", "dist_local_swing", "dist_range_mid_atr",
    "dist_ma100_atr", "zscore_price_50", "zscore_price_200"
]

# 2. Features to move to CONTINUOUS_TRIGGER_COLS and remove from RIDGE_FEATURE_META
trig_features = ["wick_to_range", "volume_spike", "orderflow_imbalance"]

# 3. Features to just remove from RIDGE_FEATURE_META
del_features = ["price_gt_ema50", "ema20_slope", "ema200_slope", "ema_slope", "bars_since_trend_flip"]

# 4. Features to add to RIDGE_FEATURE_META
add_features = {
    "volume_trend_48": '{"family": "liquidity", "type": "continuous"}',
    "volume_autocorr_48": '{"family": "liquidity", "type": "continuous"}',
    "volatility_of_volatility_48": '{"family": "volatility", "type": "continuous"}',
    "trend_acceleration": '{"family": "trend", "type": "continuous"}',
    "volatility_autocorr_48": '{"family": "volatility", "type": "continuous"}',
}

# Remove features from RIDGE_FEATURE_META
lines = content.split('\n')
new_lines = []
in_ridge = False
for line in lines:
    if line.startswith('RIDGE_FEATURE_META = {'):
        in_ridge = True
        new_lines.append(line)
        continue

    if in_ridge and line.startswith('}'):
        # Add new features before closing
        for k, v in add_features.items():
            new_lines.append(f'    "{k}": {v},')
        in_ridge = False
        new_lines.append(line)
        continue

    if in_ridge:
        # Check if line contains any feature to remove
        should_remove = False
        for feat in loc_features + trig_features + del_features:
            if f'"{feat}":' in line:
                should_remove = True
                break
        if not should_remove:
            new_lines.append(line)
    else:
        new_lines.append(line)

content = '\n'.join(new_lines)

# Add to CONTINUOUS_LOCATION_COLS
loc_block_match = re.search(r'CONTINUOUS_LOCATION_COLS = \[(.*?)\]', content, re.DOTALL)
if loc_block_match:
    loc_block = loc_block_match.group(1)
    existing_locs = [x.strip().strip('"').strip("'") for x in loc_block.split(',') if x.strip()]

    for f in loc_features:
        if f not in existing_locs:
            existing_locs.append(f)

    new_loc_block = 'CONTINUOUS_LOCATION_COLS = [\n' + ',\n'.join([f'    "{f}"' for f in existing_locs]) + ',\n]'
    content = content[:loc_block_match.start()] + new_loc_block + content[loc_block_match.end():]

# Add to CONTINUOUS_TRIGGER_COLS
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
