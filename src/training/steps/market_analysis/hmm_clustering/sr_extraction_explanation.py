#!/usr/bin/env python3
"""
Support/Resistance (SR) Feature Extraction Explanation

This script explains how the SR-specific feature extraction works in the enhanced HMM clustering system.
"""

def explain_sr_extraction():
    """Explain the Support/Resistance feature extraction process"""
    
    print("🎯 SUPPORT/RESISTANCE (SR) FEATURE EXTRACTION")
    print("="*60)
    
    print("\n📊 OVERVIEW:")
    print("The SR feature extraction creates 20 features that capture key support and resistance levels,")
    print("distances to these levels, and their strength. These features are crucial for regime detection")
    print("as they identify price zones where significant market behavior changes occur.")
    
    print("\n🔧 IMPLEMENTATION DETAILS:")
    print("="*40)
    
    print("\n1. PIVOT POINT CALCULATION:")
    print("   Formula: pivot_point = (high + low + close) / 3")
    print("   Purpose: Central reference point for SR level calculations")
    print("   Features created: 1")
    
    print("\n2. SUPPORT/RESISTANCE LEVELS:")
    print("   Support 1: 2 * pivot_point - high")
    print("   Resistance 1: 2 * pivot_point - low")
    print("   Support 2: pivot_point - (high - low)")
    print("   Resistance 2: pivot_point + (high - low)")
    print("   Features created: 4")
    
    print("\n3. DISTANCE TO S/R LEVELS:")
    print("   Distance to Support: (close - support_1) / close")
    print("   Distance to Resistance: (resistance_1 - close) / close")
    print("   Purpose: Measures how close price is to key levels")
    print("   Features created: 2")
    
    print("\n4. S/R STRENGTH CALCULATION:")
    print("   Method: _calculate_sr_strength()")
    print("   Algorithm:")
    print("     - high_swing = high.rolling(window=20, center=True).max()")
    print("     - low_swing = low.rolling(window=20, center=True).min()")
    print("     - high_strength = (high_swing - current_price) / high_swing")
    print("     - low_strength = (current_price - low_swing) / low_swing")
    print("     - sr_strength = (high_strength + low_strength) / 2")
    print("   Features created: 1")
    
    print("\n5. SWING HIGHS AND LOWS:")
    print("   Windows: [10, 20, 50] periods")
    print("   Swing High: high.rolling(window, center=True).max()")
    print("   Swing Low: low.rolling(window, center=True).min()")
    print("   Distance to Swing High: (swing_high - close) / close")
    print("   Distance to Swing Low: (close - swing_low) / close")
    print("   Features created: 12 (3 windows × 4 features)")
    
    print("\n📈 FEATURE BREAKDOWN:")
    print("="*40)
    
    feature_breakdown = {
        "Pivot Points": 1,
        "Support/Resistance Levels": 4,
        "Distance to S/R": 2,
        "S/R Strength": 1,
        "Swing Highs/Lows": 12
    }
    
    total_sr_features = 0
    for category, count in feature_breakdown.items():
        print(f"   {category}: {count} features")
        total_sr_features += count
    
    print(f"\n   TOTAL SR FEATURES: {total_sr_features}")
    
    print("\n🎯 REGIME DETECTION SIGNIFICANCE:")
    print("="*40)
    
    print("\n1. SUPPORT/RESISTANCE ZONES:")
    print("   - Identify key price levels where market behavior changes")
    print("   - Help distinguish between trending and ranging regimes")
    print("   - Provide context for volatility and momentum patterns")
    
    print("\n2. DISTANCE METRICS:")
    print("   - Close to support: Potential bounce/reversal regime")
    print("   - Close to resistance: Potential rejection/continuation regime")
    print("   - Far from levels: Trending regime with momentum")
    
    print("\n3. SWING ANALYSIS:")
    print("   - Multiple timeframes capture different regime scales")
    print("   - Short-term swings (10): Intraday regime changes")
    print("   - Medium-term swings (20): Daily regime patterns")
    print("   - Long-term swings (50): Weekly regime trends")
    
    print("\n4. STRENGTH INDICATORS:")
    print("   - High strength: Strong S/R levels, potential regime boundaries")
    print("   - Low strength: Weak levels, trending regime likely")
    print("   - Changing strength: Regime transition signals")
    
    print("\n🔗 INTERACTION WITH OTHER FEATURES:")
    print("="*40)
    
    print("\n1. VOLATILITY INTERACTIONS:")
    print("   - High volatility near S/R: Breakout regime")
    print("   - Low volatility near S/R: Consolidation regime")
    print("   - Volatility-SR interactions: 9 features")
    
    print("\n2. MOMENTUM INTERACTIONS:")
    print("   - Momentum near support: Bounce regime")
    print("   - Momentum near resistance: Rejection regime")
    print("   - Momentum-SR interactions: 9 features")
    
    print("\n3. VOLUME INTERACTIONS:")
    print("   - High volume at S/R: Institutional regime")
    print("   - Low volume at S/R: Retail regime")
    print("   - Volume-SR interactions: 9 features")
    
    print("\n📊 REGIME INTERPRETATION EXAMPLES:")
    print("="*40)
    
    regime_examples = {
        "Bull Breakout": {
            "sr_context": "Price breaks above resistance with high volume",
            "features": "distance_to_resistance < 0.01, volume_spike = True",
            "regime_type": "bull_breakout"
        },
        "Bear Breakdown": {
            "sr_context": "Price breaks below support with high volume",
            "features": "distance_to_support < 0.01, volume_spike = True",
            "regime_type": "bear_breakdown"
        },
        "Support Bounce": {
            "sr_context": "Price bounces off support level",
            "features": "distance_to_support < 0.005, momentum_5 > 0",
            "regime_type": "support_bounce"
        },
        "Resistance Rejection": {
            "sr_context": "Price rejects at resistance level",
            "features": "distance_to_resistance < 0.005, momentum_5 < 0",
            "regime_type": "resistance_rejection"
        },
        "Consolidation": {
            "sr_context": "Price oscillates between support and resistance",
            "features": "sr_strength > 0.7, volatility_20 < 0.02",
            "regime_type": "consolidation"
        }
    }
    
    for regime, details in regime_examples.items():
        print(f"\n{regime.upper()}:")
        print(f"   Context: {details['sr_context']}")
        print(f"   Key Features: {details['features']}")
        print(f"   Regime Type: {details['regime_type']}")
    
    print("\n🚀 ADVANCED SR FEATURES:")
    print("="*40)
    
    print("\n1. BOUNCE SIGNALS:")
    print("   - support_bounce_signal: Detects price bounces off support")
    print("   - resistance_bounce_signal: Detects price rejections at resistance")
    print("   - Algorithm: Price approaches level, then reverses direction")
    
    print("\n2. PROXIMITY INDICATORS:")
    print("   - near_support: Binary indicator (within 0.5% of support)")
    print("   - near_resistance: Binary indicator (within 0.5% of resistance)")
    print("   - Purpose: Quick regime classification")
    
    print("\n3. REGIME-SPECIFIC SR METRICS:")
    print("   - sr_proximity_by_regime: S/R proximity for each HMM state")
    print("   - sr_strength_by_regime: S/R strength for each HMM state")
    print("   - overall_sr_metrics: Aggregate S/R context")
    
    print("\n💡 KEY INSIGHTS:")
    print("="*40)
    
    print("\n1. REGIME BOUNDARIES:")
    print("   - S/R levels act as natural regime boundaries")
    print("   - Price behavior changes significantly near these levels")
    print("   - Distance metrics provide regime transition signals")
    
    print("\n2. MULTI-TIMEFRAME ANALYSIS:")
    print("   - 10-period swings: Short-term regime changes")
    print("   - 20-period swings: Medium-term regime patterns")
    print("   - 50-period swings: Long-term regime trends")
    
    print("\n3. STRENGTH-BASED REGIME CLASSIFICATION:")
    print("   - High S/R strength: Ranging/consolidation regime")
    print("   - Low S/R strength: Trending regime")
    print("   - Changing strength: Regime transition")
    
    print("\n4. INTERACTION ENRICHMENT:")
    print("   - 27 additional features from S/R interactions")
    print("   - Captures regime-specific S/R behavior")
    print("   - Enhances regime discrimination accuracy")
    
    print(f"\n🎯 TOTAL SR CONTRIBUTION: {total_sr_features} base features + 27 interaction features = {total_sr_features + 27} total SR-related features")

if __name__ == "__main__":
    explain_sr_extraction()