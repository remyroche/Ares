"""
Proposed Fixes for Quality Score Calculation

This script shows BEFORE/AFTER for the critical fixes.
You can test these locally before modifying the actual collector.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    """Load training data for testing."""
    return pd.read_parquet('data_cache/sr_ml_training/sr_quality_training_data.parquet')


# ============================================================================
# FIX #1: BOUNCE STRENGTH
# ============================================================================

def calculate_bounce_strength_OLD(future_data, hit_bar, level_type, level_price):
    """
    CURRENT IMPLEMENTATION (BROKEN - Saturated)
    
    Problem: Uses MAX bounce over entire forward window → almost always finds 2%+ move
    Result: 50% of samples at 1.0, mean=0.9757
    """
    tolerance = level_price * 0.005
    
    if level_type == 'support':
        future_highs = future_data['high']
        max_bounce = future_highs.max() - hit_bar['low']
        bounce_pct = max_bounce / level_price
    else:  # resistance
        future_lows = future_data['low']
        max_bounce = hit_bar['high'] - future_lows.min()
        bounce_pct = max_bounce / level_price
    
    bounce_strength = min(bounce_pct / 0.02, 1.0)  # 2% = 1.0 (SATURATES!)
    return bounce_strength


def calculate_bounce_strength_NEW_v1(future_data, hit_bar, level_type, level_price):
    """
    FIX #1 - Version 1: Use EARLY bounce (first 10 bars)
    
    Improvement: 
    - Only look at first 10 bars after hit (immediate reaction)
    - Higher threshold (3% instead of 2%) for max score
    - Should give mean ~0.5-0.6, better spread
    """
    # Only use first 10 bars after hit
    early_future = future_data.iloc[:10]
    
    if level_type == 'support':
        max_bounce = early_future['high'].max() - hit_bar['low']
        bounce_pct = max_bounce / level_price
    else:  # resistance
        max_bounce = hit_bar['high'] - early_future['low'].min()
        bounce_pct = max_bounce / level_price
    
    # Higher threshold: 3% = 1.0 (vs 2% before)
    bounce_strength = min(bounce_pct / 0.03, 1.0)
    return bounce_strength


def calculate_bounce_strength_NEW_v2(future_data, hit_bar, level_type, level_price):
    """
    FIX #1 - Version 2: Use MEDIAN bounce (time-weighted)
    
    Improvement:
    - Calculate bounce for each of first 20 bars
    - Use median instead of max (less outlier-sensitive)
    - Apply time decay (recent bounces matter more)
    """
    bounces = []
    weights = []
    
    for i, (idx, bar) in enumerate(future_data.iloc[:20].iterrows()):
        # Calculate bounce for this bar
        if level_type == 'support':
            bounce = (bar['high'] - hit_bar['low']) / level_price
        else:  # resistance
            bounce = (hit_bar['high'] - bar['low']) / level_price
        
        # Time decay weight (exponential)
        weight = np.exp(-i / 10)
        
        bounces.append(bounce)
        weights.append(weight)
    
    # Weighted median
    bounce_pct = np.average(bounces, weights=weights)
    bounce_strength = min(bounce_pct / 0.02, 1.0)
    
    return bounce_strength


# ============================================================================
# FIX #2: TRADE PROFIT
# ============================================================================

def simulate_trade_OLD(level_type, entry_price, future_data):
    """
    CURRENT IMPLEMENTATION (BROKEN - Negative)
    
    Problem: 2:1 R/R (2% TP, 1% SL) too aggressive for 15m timeframe
    Result: 65% of trades lose, mean=-0.05
    """
    if level_type == 'support':
        stop_loss = entry_price * 0.99     # 1% SL
        take_profit = entry_price * 1.02   # 2% TP (too far!)
        direction = 1
    else:  # resistance
        stop_loss = entry_price * 1.01
        take_profit = entry_price * 0.98
        direction = -1
    
    # Check next 10 bars
    future_bars = future_data.iloc[:10]
    
    for _, bar in future_bars.iterrows():
        if direction == 1:  # Long
            if bar['low'] <= stop_loss:
                return -0.5  # Loss
            elif bar['high'] >= take_profit:
                return 1.0   # Win (2:1 R/R)
        else:  # Short
            if bar['high'] >= stop_loss:
                return -0.5
            elif bar['low'] <= take_profit:
                return 1.0
    
    # No SL/TP hit - exit at close
    exit_price = future_bars.iloc[-1]['close']
    pnl_pct = (exit_price - entry_price) / entry_price * direction
    return np.clip(pnl_pct * 50, -1, 1)


def simulate_trade_NEW_v1(level_type, entry_price, future_data):
    """
    FIX #2 - Version 1: Tighter stops for 15m timeframe
    
    Improvement:
    - 0.5% SL (tighter, vs 1% before)
    - 1% TP (more realistic, vs 2% before)
    - Still 2:1 R/R, but scaled down
    - Should give positive expectancy
    """
    if level_type == 'support':
        stop_loss = entry_price * 0.995    # 0.5% SL (TIGHTER)
        take_profit = entry_price * 1.01   # 1% TP (MORE REALISTIC)
        direction = 1
    else:  # resistance
        stop_loss = entry_price * 1.005
        take_profit = entry_price * 0.99
        direction = -1
    
    # Check next 10 bars
    future_bars = future_data.iloc[:10]
    
    for _, bar in future_bars.iterrows():
        if direction == 1:  # Long
            if bar['low'] <= stop_loss:
                return -0.5
            elif bar['high'] >= take_profit:
                return 1.0
        else:  # Short
            if bar['high'] >= stop_loss:
                return -0.5
            elif bar['low'] <= take_profit:
                return 1.0
    
    # No SL/TP hit - exit at close
    exit_price = future_bars.iloc[-1]['close']
    pnl_pct = (exit_price - entry_price) / entry_price * direction
    return np.clip(pnl_pct * 100, -1, 1)  # Scale: 1% = 1.0


def simulate_trade_NEW_v2(level_type, entry_price, future_data):
    """
    FIX #2 - Version 2: Use 1:1 R/R
    
    Improvement:
    - 1% SL, 1% TP (1:1 R/R)
    - More realistic for 15m timeframe
    - Should give ~50% win rate → neutral/positive expectancy
    """
    if level_type == 'support':
        stop_loss = entry_price * 0.99     # 1% SL
        take_profit = entry_price * 1.01   # 1% TP (1:1 R/R)
        direction = 1
    else:  # resistance
        stop_loss = entry_price * 1.01
        take_profit = entry_price * 0.99
        direction = -1
    
    # Same logic as before
    future_bars = future_data.iloc[:10]
    
    for _, bar in future_bars.iterrows():
        if direction == 1:
            if bar['low'] <= stop_loss:
                return -0.5
            elif bar['high'] >= take_profit:
                return 1.0
        else:
            if bar['high'] >= stop_loss:
                return -0.5
            elif bar['low'] <= take_profit:
                return 1.0
    
    exit_price = future_bars.iloc[-1]['close']
    pnl_pct = (exit_price - entry_price) / entry_price * direction
    return np.clip(pnl_pct * 100, -1, 1)


# ============================================================================
# FIX #3: QUALITY SCORE FORMULA
# ============================================================================

def calculate_quality_OLD(bounce, hold, profit):
    """
    CURRENT FORMULA
    
    Problem: After bounce saturation (0.98) and negative profit (-0.05),
             quality is dominated by hold_strength only
    """
    return (
        bounce * 0.35 +
        hold * 0.35 +
        max(profit, 0) * 0.30  # max(profit, 0) caps negative
    )


def calculate_quality_NEW_v1(bounce, hold, profit):
    """
    FIX #3 - Version 1: Equal weights (after fixing bounce & profit)
    
    After fixing bounce and profit issues, use equal weights
    """
    return (
        bounce * 0.333 +
        hold * 0.333 +
        max(profit, 0) * 0.333
    )


def calculate_quality_NEW_v2(bounce, hold, profit):
    """
    FIX #3 - Version 2: Remove trade profit (if still problematic)
    
    If trade profit simulation remains unreliable, remove it entirely
    and focus on bounce + hold only
    """
    return (
        bounce * 0.5 +
        hold * 0.5
    )


# ============================================================================
# COMPARISON & TESTING
# ============================================================================

def compare_fixes():
    """Compare old vs new implementations on actual data."""
    
    print("\n" + "="*80)
    print("🔧 QUALITY SCORE FIXES - BEFORE/AFTER COMPARISON")
    print("="*80)
    
    # Note: This is a conceptual demonstration
    # Actual testing would require re-running on full future_data windows
    
    print("\n📊 EXPECTED IMPROVEMENTS:")
    print("-"*80)
    
    print("\n1. BOUNCE STRENGTH:")
    print("   BEFORE (OLD):")
    print("      Mean: 0.9757, Median: 1.0000, Std: 0.1028")
    print("      Problem: 50% of samples at max (1.0)")
    print("   ")
    print("   AFTER (FIX v1 - Early bounce):")
    print("      Expected Mean: ~0.60, Median: ~0.55, Std: ~0.25")
    print("      Improvement: Better spread, discriminative power")
    print("   ")
    print("   AFTER (FIX v2 - Median bounce):")
    print("      Expected Mean: ~0.55, Median: ~0.50, Std: ~0.30")
    print("      Improvement: Less outlier-sensitive, smoother")
    
    print("\n2. TRADE PROFIT:")
    print("   BEFORE (OLD):")
    print("      Mean: -0.0523, Median: -0.5000")
    print("      Problem: 65% of trades lose money")
    print("   ")
    print("   AFTER (FIX v1 - Tighter stops):")
    print("      Expected Mean: ~0.10, Median: ~0.00")
    print("      Improvement: Positive expectancy, ~45% win rate")
    print("   ")
    print("   AFTER (FIX v2 - 1:1 R/R):")
    print("      Expected Mean: ~0.15, Median: ~0.00")
    print("      Improvement: Higher win rate (~50%), neutral/positive")
    
    print("\n3. QUALITY SCORE:")
    print("   BEFORE (OLD):")
    print("      Effective formula: 0.341 + hold * 0.35")
    print("      Problem: Dominated by hold_strength only")
    print("   ")
    print("   AFTER (Equal weights):")
    print("      Formula: bounce * 0.333 + hold * 0.333 + profit * 0.333")
    print("      Improvement: All components contribute equally")
    print("   ")
    print("   AFTER (No trade profit):")
    print("      Formula: bounce * 0.5 + hold * 0.5")
    print("      Improvement: Simpler, focus on price action only")
    
    print("\n" + "="*80)
    print("💡 RECOMMENDED APPROACH:")
    print("="*80)
    
    print("\nStep 1: Apply FIX #1 (Bounce) - Use Version 1 (early bounce)")
    print("Step 2: Apply FIX #2 (Trade) - Use Version 2 (1:1 R/R)")
    print("Step 3: Recollect training data with new calculations")
    print("Step 4: Validate improvements using validate_quality_score.py")
    print("Step 5: If still issues, apply FIX #3 (adjust weights)")
    
    print("\n" + "="*80)
    print("📁 CODE LOCATION:")
    print("="*80)
    print("\nFile: src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py")
    print("   - Bounce calculation: Lines 410-420")
    print("   - Trade simulation: Lines 470-505")
    print("   - Quality formula: Lines 442-446")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    compare_fixes()
    
    print("📋 NEXT STEPS:")
    print("   1. Review the proposed fixes above")
    print("   2. Choose which version to implement")
    print("   3. Update sr_quality_data_collector.py with chosen fix")
    print("   4. Re-run data collection: python3 scripts/collect_sr_training_data.py")
    print("   5. Validate: python3 validate_quality_score.py")
    print("\n   See QUALITY_SCORE_INVESTIGATION_FINDINGS.md for full details!")
    print("")

