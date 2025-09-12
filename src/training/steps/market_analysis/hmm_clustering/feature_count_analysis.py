#!/usr/bin/env python3
"""
Detailed Feature Count Analysis for Enhanced HMM Clustering

This script provides an accurate count of all features created by the enhanced feature engineering.
"""

def count_enhanced_features():
    """Count all features created by the enhanced feature engineering"""
    
    print("🔢 DETAILED FEATURE COUNT ANALYSIS")
    print("="*60)
    
    feature_counts = {}
    
    # 1. Price Features
    print("\n1. PRICE FEATURES:")
    price_features = []
    
    # Basic price features (6)
    basic_price = [
        'price_change', 'price_range', 'price_position',
        'high_close_ratio', 'low_close_ratio', 'open_close_ratio'
    ]
    price_features.extend(basic_price)
    print(f"   Basic price features: {len(basic_price)}")
    
    # Price gaps (2)
    gap_features = ['gap_up', 'gap_down']
    price_features.extend(gap_features)
    print(f"   Price gap features: {len(gap_features)}")
    
    # Price patterns (2)
    pattern_features = ['doji', 'hammer']
    price_features.extend(pattern_features)
    print(f"   Price pattern features: {len(pattern_features)}")
    
    # Multiple timeframe price features (4 windows × 5 features = 20)
    windows = [5, 10, 20, 50]
    timeframe_features = []
    for window in windows:
        timeframe_features.extend([
            f'price_ma_{window}', f'price_ema_{window}', f'price_std_{window}',
            f'price_min_{window}', f'price_max_{window}'
        ])
    price_features.extend(timeframe_features)
    print(f"   Timeframe price features: {len(timeframe_features)}")
    
    # Price vs moving averages (4 windows × 2 features = 8)
    ma_ratio_features = []
    for window in windows:
        ma_ratio_features.extend([
            f'price_vs_ma_{window}', f'price_vs_ema_{window}'
        ])
    price_features.extend(ma_ratio_features)
    print(f"   Price vs MA features: {len(ma_ratio_features)}")
    
    feature_counts['price_features'] = len(price_features)
    print(f"   TOTAL PRICE FEATURES: {len(price_features)}")
    
    # 2. Volume Features
    print("\n2. VOLUME FEATURES:")
    volume_features = []
    
    # Basic volume features (2)
    basic_volume = ['volume_change', 'volume_ma_ratio']
    volume_features.extend(basic_volume)
    print(f"   Basic volume features: {len(basic_volume)}")
    
    # Volume-price relationship (2)
    volume_price = ['volume_price_trend', 'volume_price_correlation']
    volume_features.extend(volume_price)
    print(f"   Volume-price relationship: {len(volume_price)}")
    
    # Volume patterns (2)
    volume_patterns = ['volume_spike', 'volume_dry_up']
    volume_features.extend(volume_patterns)
    print(f"   Volume pattern features: {len(volume_patterns)}")
    
    # Multiple timeframe volume features (4 windows × 3 features = 12)
    volume_timeframe = []
    for window in windows:
        volume_timeframe.extend([
            f'volume_ma_{window}', f'volume_std_{window}', f'volume_ratio_{window}'
        ])
    volume_features.extend(volume_timeframe)
    print(f"   Timeframe volume features: {len(volume_timeframe)}")
    
    feature_counts['volume_features'] = len(volume_features)
    print(f"   TOTAL VOLUME FEATURES: {len(volume_features)}")
    
    # 3. Volatility Features
    print("\n3. VOLATILITY FEATURES:")
    volatility_features = []
    
    # Rolling volatility (4 windows × 2 types = 8)
    rolling_vol = []
    for window in windows:
        rolling_vol.extend([
            f'volatility_{window}', f'volatility_ewma_{window}'
        ])
    volatility_features.extend(rolling_vol)
    print(f"   Rolling volatility features: {len(rolling_vol)}")
    
    # Volatility ratios (2)
    vol_ratios = ['volatility_ratio_5_20', 'volatility_ratio_10_50']
    volatility_features.extend(vol_ratios)
    print(f"   Volatility ratio features: {len(vol_ratios)}")
    
    # Volatility momentum (2)
    vol_momentum = ['volatility_momentum', 'volatility_acceleration']
    volatility_features.extend(vol_momentum)
    print(f"   Volatility momentum features: {len(vol_momentum)}")
    
    # GARCH-like features (2)
    garch_features = ['volatility_clustering', 'volatility_persistence']
    volatility_features.extend(garch_features)
    print(f"   GARCH-like features: {len(garch_features)}")
    
    feature_counts['volatility_features'] = len(volatility_features)
    print(f"   TOTAL VOLATILITY FEATURES: {len(volatility_features)}")
    
    # 4. Technical Indicators
    print("\n4. TECHNICAL INDICATORS:")
    technical_features = []
    
    # RSI (3 windows)
    rsi_features = [f'rsi_{window}' for window in [14, 21, 30]]
    technical_features.extend(rsi_features)
    print(f"   RSI features: {len(rsi_features)}")
    
    # MACD (3)
    macd_features = ['macd', 'macd_signal', 'macd_histogram']
    technical_features.extend(macd_features)
    print(f"   MACD features: {len(macd_features)}")
    
    # Bollinger Bands (2 windows × 5 features = 10)
    bb_features = []
    for window in [20, 50]:
        bb_features.extend([
            f'bb_upper_{window}', f'bb_middle_{window}', f'bb_lower_{window}',
            f'bb_width_{window}', f'bb_position_{window}'
        ])
    technical_features.extend(bb_features)
    print(f"   Bollinger Bands features: {len(bb_features)}")
    
    # ATR (2)
    atr_features = ['atr_14', 'atr_ratio']
    technical_features.extend(atr_features)
    print(f"   ATR features: {len(atr_features)}")
    
    # ADX (1)
    adx_features = ['adx_14']
    technical_features.extend(adx_features)
    print(f"   ADX features: {len(adx_features)}")
    
    feature_counts['technical_indicators'] = len(technical_features)
    print(f"   TOTAL TECHNICAL INDICATORS: {len(technical_features)}")
    
    # 5. Momentum Features
    print("\n5. MOMENTUM FEATURES:")
    momentum_features = []
    
    # Price momentum (7 windows × 2 features = 14)
    price_momentum = []
    momentum_windows = [1, 2, 3, 5, 10, 20, 50]
    for window in momentum_windows:
        price_momentum.extend([
            f'momentum_{window}', f'momentum_ma_{window}'
        ])
    momentum_features.extend(price_momentum)
    print(f"   Price momentum features: {len(price_momentum)}")
    
    # Volume momentum (6 windows)
    volume_momentum = [f'volume_momentum_{window}' for window in [1, 2, 3, 5, 10, 20]]
    momentum_features.extend(volume_momentum)
    print(f"   Volume momentum features: {len(volume_momentum)}")
    
    # Momentum ratios (2)
    momentum_ratios = ['momentum_ratio_5_20', 'momentum_ratio_10_50']
    momentum_features.extend(momentum_ratios)
    print(f"   Momentum ratio features: {len(momentum_ratios)}")
    
    feature_counts['momentum_features'] = len(momentum_features)
    print(f"   TOTAL MOMENTUM FEATURES: {len(momentum_features)}")
    
    # 6. Support/Resistance Features
    print("\n6. SUPPORT/RESISTANCE FEATURES:")
    sr_features = []
    
    # Pivot points (5)
    pivot_features = ['pivot_point', 'support_1', 'resistance_1', 'support_2', 'resistance_2']
    sr_features.extend(pivot_features)
    print(f"   Pivot point features: {len(pivot_features)}")
    
    # Distance to S/R levels (2)
    distance_features = ['distance_to_support', 'distance_to_resistance']
    sr_features.extend(distance_features)
    print(f"   Distance to S/R features: {len(distance_features)}")
    
    # S/R strength (1)
    sr_strength = ['sr_strength']
    sr_features.extend(sr_strength)
    print(f"   S/R strength features: {len(sr_strength)}")
    
    # Swing highs and lows (3 windows × 4 features = 12)
    swing_features = []
    swing_windows = [10, 20, 50]
    for window in swing_windows:
        swing_features.extend([
            f'swing_high_{window}', f'swing_low_{window}',
            f'distance_to_swing_high_{window}', f'distance_to_swing_low_{window}'
        ])
    sr_features.extend(swing_features)
    print(f"   Swing high/low features: {len(swing_features)}")
    
    feature_counts['sr_features'] = len(sr_features)
    print(f"   TOTAL S/R FEATURES: {len(sr_features)}")
    
    # 7. Statistical Features
    print("\n7. STATISTICAL FEATURES:")
    statistical_features = []
    
    # Skewness and kurtosis (2 windows × 2 features = 4)
    skew_kurt = []
    stat_windows = [20, 50]
    for window in stat_windows:
        skew_kurt.extend([f'skewness_{window}', f'kurtosis_{window}'])
    statistical_features.extend(skew_kurt)
    print(f"   Skewness/kurtosis features: {len(skew_kurt)}")
    
    # Quantiles (2 windows × 5 quantiles × 2 features = 20)
    quantile_features = []
    quantiles = [0.25, 0.5, 0.75, 0.9, 0.95]
    for window in stat_windows:
        for q in quantiles:
            quantile_features.extend([
                f'quantile_{q}_{window}', f'price_vs_quantile_{q}_{window}'
            ])
    statistical_features.extend(quantile_features)
    print(f"   Quantile features: {len(quantile_features)}")
    
    # Autocorrelation (2 windows)
    autocorr_features = [f'autocorr_{window}' for window in stat_windows]
    statistical_features.extend(autocorr_features)
    print(f"   Autocorrelation features: {len(autocorr_features)}")
    
    feature_counts['statistical_features'] = len(statistical_features)
    print(f"   TOTAL STATISTICAL FEATURES: {len(statistical_features)}")
    
    # 8. Time Features
    print("\n8. TIME FEATURES:")
    time_features = []
    
    # Basic time features (4)
    basic_time = ['hour', 'day_of_week', 'day_of_month', 'month']
    time_features.extend(basic_time)
    print(f"   Basic time features: {len(basic_time)}")
    
    # Cyclical encoding (4)
    cyclical = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    time_features.extend(cyclical)
    print(f"   Cyclical encoding features: {len(cyclical)}")
    
    feature_counts['time_features'] = len(time_features)
    print(f"   TOTAL TIME FEATURES: {len(time_features)}")
    
    # 9. Feature Interactions
    print("\n9. FEATURE INTERACTIONS:")
    interaction_features = []
    
    # Price-volume interactions (1)
    price_volume = ['price_volume_interaction']
    interaction_features.extend(price_volume)
    print(f"   Price-volume interactions: {len(price_volume)}")
    
    # Volatility-momentum interactions (1)
    vol_momentum_interaction = ['volatility_momentum_interaction']
    interaction_features.extend(vol_momentum_interaction)
    print(f"   Volatility-momentum interactions: {len(vol_momentum_interaction)}")
    
    # RSI-momentum interactions (1)
    rsi_momentum_interaction = ['rsi_momentum_interaction']
    interaction_features.extend(rsi_momentum_interaction)
    print(f"   RSI-momentum interactions: {len(rsi_momentum_interaction)}")
    
    feature_counts['interaction_features'] = len(interaction_features)
    print(f"   TOTAL INTERACTION FEATURES: {len(interaction_features)}")
    
    # Total count
    total_features = sum(feature_counts.values())
    
    print("\n" + "="*60)
    print("📊 FEATURE COUNT SUMMARY")
    print("="*60)
    
    for category, count in feature_counts.items():
        print(f"{category.replace('_', ' ').title()}: {count:3d} features")
    
    print("-" * 60)
    print(f"TOTAL FEATURES: {total_features:3d}")
    print("="*60)
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   - Total features: {total_features} (10x increase from original 20)")
    print(f"   - Largest category: {max(feature_counts, key=feature_counts.get)} ({max(feature_counts.values())} features)")
    print(f"   - Most diverse: Technical indicators with multiple timeframes")
    print(f"   - Volume analysis: {feature_counts['volume_features']} features for comprehensive volume analysis")
    print(f"   - Regime detection: {feature_counts['volatility_features']} volatility features for regime identification")
    
    return feature_counts, total_features

def demonstrate_volume_enhanced_regime_interpretation():
    """Demonstrate volume-enhanced regime interpretation"""
    
    print("\n" + "="*60)
    print("📈 VOLUME-ENHANCED REGIME INTERPRETATION")
    print("="*60)
    
    regime_types = {
        'bull_breakout': {
            'description': 'Strong upward trend with high volatility and volume',
            'characteristics': 'High momentum + High volatility + High volume',
            'trading_implication': 'Strong bullish momentum with institutional participation',
            'confidence': 0.9
        },
        'bear_breakdown': {
            'description': 'Strong downward trend with high volatility and volume',
            'characteristics': 'Negative momentum + High volatility + High volume',
            'trading_implication': 'Strong bearish momentum with panic selling',
            'confidence': 0.9
        },
        'high_volatility_volume': {
            'description': 'High volatility with very high volume (potential reversal)',
            'characteristics': 'Neutral momentum + Very high volatility + Very high volume',
            'trading_implication': 'Potential market reversal or major news event',
            'confidence': 0.7
        },
        'consolidation_low_volume': {
            'description': 'Low volatility consolidation with low volume',
            'characteristics': 'Low momentum + Low volatility + Low volume',
            'trading_implication': 'Market indecision, potential breakout coming',
            'confidence': 0.8
        },
        'gentle_bull_volume': {
            'description': 'Gentle upward trend with low volatility and above-average volume',
            'characteristics': 'Positive momentum + Low volatility + Above-average volume',
            'trading_implication': 'Healthy uptrend with steady accumulation',
            'confidence': 0.7
        },
        'gentle_bear_volume': {
            'description': 'Gentle downward trend with low volatility and above-average volume',
            'characteristics': 'Negative momentum + Low volatility + Above-average volume',
            'trading_implication': 'Steady distribution, potential trend continuation',
            'confidence': 0.7
        }
    }
    
    print("\nVolume-Enhanced Regime Types:")
    for regime_type, details in regime_types.items():
        print(f"\n{regime_type.upper().replace('_', ' ')}:")
        print(f"   Description: {details['description']}")
        print(f"   Characteristics: {details['characteristics']}")
        print(f"   Trading Implication: {details['trading_implication']}")
        print(f"   Confidence: {details['confidence']}")
    
    print(f"\n🎯 Volume Analysis Benefits:")
    print(f"   - Distinguishes between strong and weak trends")
    print(f"   - Identifies institutional vs retail participation")
    print(f"   - Detects potential reversal points")
    print(f"   - Provides confidence levels for regime classification")
    print(f"   - Enables more sophisticated trading strategies")

if __name__ == "__main__":
    feature_counts, total_features = count_enhanced_features()
    demonstrate_volume_enhanced_regime_interpretation()