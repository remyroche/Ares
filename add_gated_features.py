"""
Add Gated Entry Features for LONG_MR Enhancement

This script adds 4 new features to features.py:
1. bounce_signal: close[t] > close[t-1] AND extreme move detected
2. trap_strength: custom trap quality metric (encapsulates trap characteristics)
3. volume_capitulation: volume > 2× average
4. entry_quality_composite: combines all 3 conditions

These will be added to the feature generation pipeline.
"""

# Feature implementations to add to _compute_features_impl in features.py

def add_gated_entry_features():
    """
    Code to add after existing feature calculations in _compute_features_impl
    """
    code = '''
    # ========================================================================
    # GATED ENTRY FEATURES (for LONG_MR enhancement)
    # ========================================================================
    
    # 1. Bounce Signal: close[t] > close[t-1] AND extreme move detected
    close_bounce = (c > c.shift(1)).astype(float)
    extreme_move = (np.abs(ret1h) > atr_pct * 2.0).astype(float)  # 2× ATR move
    feats["bounce_signal"] = close_bounce * extreme_move
    
    # 2. Trap Strength: Custom trap quality metric
    # Combines: price at extreme + volume spike + reversal pattern
    price_at_low = ((l - l.rolling(12).min()) / (h.rolling(12).max() - l.rolling(12).min() + 1e-9))
    price_at_high = ((h.rolling(12).max() - h) / (h.rolling(12).max() - l.rolling(12).min() + 1e-9))
    
    vol_spike = (v / v.rolling(24).mean()).clip(upper=5.0)  # Volume relative to 24h avg
    
    # For longs: trap at bottom (price_at_low close to 0) + volume spike
    # For shorts: trap at top (price_at_high close to 0) + volume spike
    trap_long = (1.0 - price_at_low) * (vol_spike / 5.0)  # Normalize to [0,1]
    trap_short = (1.0 - price_at_high) * (vol_spike / 5.0)
    
    # Use trend to determine which trap to use
    trend_sign = np.sign(trend_pct)
    feats["trap_strength"] = np.where(
        trend_sign >= 0,
        trap_long,  # Uptrend: use long trap (buy dips)
        trap_short  # Downtrend: use short trap (sell rips)
    )
    
    # 3. Volume Capitulation: volume > 2× average
    vol_ma_24 = v.rolling(24).mean()
    feats["volume_capitulation"] = (v > vol_ma_24 * 2.0).astype(float)
    
    # 4. Entry Quality Composite: combines all 3 conditions
    # Weighted combination: 40% bounce + 30% trap + 30% volume
    feats["entry_quality_composite"] = (
        0.4 * feats["bounce_signal"] +
        0.3 * feats["trap_strength"] +
        0.3 * feats["volume_capitulation"]
    )
    
    # Add to feature list for validation
    check_inf_nan(feats, "bounce_signal")
    check_inf_nan(feats, "trap_strength")
    check_inf_nan(feats, "volume_capitulation")
    check_inf_nan(feats, "entry_quality_composite")
    '''
    
    return code

# Location to add: After line ~800 in features.py, before final return statement
# Look for the section where other features are calculated

print("=" * 80)
print("GATED ENTRY FEATURES CODE")
print("=" * 80)
print("\nAdd this code to _compute_features_impl in features.py")
print("Location: After existing feature calculations, before 'return feats'\n")
print(add_gated_entry_features())
print("\n" + "=" * 80)
print("CONFIGURATION UPDATE")
print("=" * 80)
print("\nAdd these features to config.py meta_feature_keys:")
print('"bounce_signal", "trap_strength", "volume_capitulation", "entry_quality_composite"')
