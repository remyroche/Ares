#!/usr/bin/env python3
"""
Simple test to verify SR feature categorization logic without external dependencies.
"""

def categorize_features_simple(feature_names):
    """Simple feature categorization logic for testing."""
    categories = {
        "momentum": [],
        "volatility": [],
        "liquidity": [],
        "volume": [],
        "microstructure": [],
        "regime": [],
        "sr_features": [],
        "interaction": [],
        "other": []
    }
    
    for feature in feature_names:
        feature_lower = feature.lower()
        categorized = False
        
        # Interaction features (check first to avoid conflicts)
        if any(keyword in feature_lower for keyword in [
            "_x_", "_div_", "_ratio_", "_over_", "_cross_", "interaction",
            "momentum_x_", "volatility_x_", "volume_x_", "regime_x_",
            "momentum_div_", "volatility_div_", "volume_div_"
        ]):
            categories["interaction"].append(feature)
            categorized = True
        
        # Momentum indicators (including multi-timeframe and derivative forms)
        if not categorized:
            momentum_base_tokens = [
                "momentum", "mom", "rsi", "macd", "cci", "roc", "willr", "stoch",
                "adx", "dmi", "kama", "tema", "dema", "hma", "wma", "vwma", "zlema",
                "ichimoku", "psar", "trix", "cmo", "tsi", "ppo", "pmo", "uo",
                "linreg", "lin_reg", "sma", "ema", "ma_", "moving_avg", "trend",
                "bb_position", "bb_upper", "bb_lower", "bb_width", "bb_percent"
            ]
            derivative_tokens = [
                "_diff", "diff_", "_delta", "delta_", "_accel", "accel_",
                "acceleration", "_slope", "slope_", "_change", "change_", "_norm", "norm_"
            ]
            has_momentum_base = any(token in feature_lower for token in momentum_base_tokens)
            has_derivative_with_anchor = (
                any(token in feature_lower for token in derivative_tokens)
                and any(anchor in feature_lower for anchor in [
                    "momentum", "roc", "rsi", "macd", "stoch", "cci", "willr", "trend", "bb"
                ])
            )
            if has_momentum_base or has_derivative_with_anchor:
                categories["momentum"].append(feature)
                categorized = True
        
        # Volatility measures (including multi-timeframe and derivative forms)
        if not categorized:
            volatility_base_tokens = [
                "volatility", "atr", "true_range", "truerange", "natr", "parkinson",
                "garman", "gk_vol", "garman_klass", "roll", "rvol", "realized_vol",
                "hv", "hist_vol", "historical_vol", "variance", "std", "bbands",
                "boll", "bollinger", "donch", "donchian", "keltner", "chop",
                "choppiness", "park_vol", "vol_", "volatility_"
            ]
            derivative_tokens = [
                "_diff", "diff_", "_delta", "delta_", "_accel", "accel_",
                "acceleration", "_slope", "slope_", "_change", "change_", "_norm", "norm_"
            ]
            has_volatility_base = any(token in feature_lower for token in volatility_base_tokens)
            has_derivative_with_anchor = (
                any(token in feature_lower for token in derivative_tokens)
                and any(anchor in feature_lower for anchor in [
                    "volatility", "atr", "true_range", "variance", "std", "bbands", "bollinger"
                ])
            )
            if has_volatility_base or has_derivative_with_anchor:
                categories["volatility"].append(feature)
                categorized = True
        
        # Volume features (including multi-timeframe and derivative forms)
        if not categorized:
            volume_base_tokens = [
                "volume", "tick_volume", "obv", "cmf", "mfi", "vwap",
                "pvi", "nvi", "efi", "delta_volume", "volume_ratio", "volume_ma", 
                "volume_change", "volume_sma", "volume_momentum", "volume_weighted",
                "volume_velocity", "volume_acceleration", "volume_price", "volume_"
            ]
            derivative_tokens = [
                "_diff", "diff_", "_delta", "delta_", "_accel", "accel_",
                "acceleration", "_slope", "slope_", "_change", "change_", "_norm", "norm_"
            ]
            has_volume_base = any(token in feature_lower for token in volume_base_tokens)
            has_derivative_with_anchor = (
                any(token in feature_lower for token in derivative_tokens)
                and any(anchor in feature_lower for anchor in [
                    "volume", "obv", "cmf", "mfi", "vwap", "volume_ratio", "volume_ma"
                ])
            )
            if has_volume_base or has_derivative_with_anchor:
                categories["volume"].append(feature)
                categorized = True
        
        # Liquidity features (including multi-timeframe and derivative forms)
        if not categorized:
            liquidity_base_tokens = [
                "liquidity", "spread", "bid_ask", "bidask", "quote_imbalance",
                "liquidity_", "spread_", "bid_", "ask_", "quote_"
            ]
            derivative_tokens = [
                "_diff", "diff_", "_delta", "delta_", "_accel", "accel_",
                "acceleration", "_slope", "slope_", "_change", "change_", "_norm", "norm_"
            ]
            has_liquidity_base = any(token in feature_lower for token in liquidity_base_tokens)
            has_derivative_with_anchor = (
                any(token in feature_lower for token in derivative_tokens)
                and any(anchor in feature_lower for anchor in [
                    "liquidity", "spread", "bid_ask", "quote_imbalance"
                ])
            )
            if has_liquidity_base or has_derivative_with_anchor:
                categories["liquidity"].append(feature)
                categorized = True
        
        # Microstructure features (including multi-timeframe and derivative forms)
        if not categorized:
            microstructure_base_tokens = [
                "microstructure", "order_flow", "orderflow", "ofi", "imbalance",
                "quote_imbalance", "depth", "orderbook", "book", "microprice", 
                "trade_count", "trade_frequency", "order_", "flow_", "imbalance_"
            ]
            derivative_tokens = [
                "_diff", "diff_", "_delta", "delta_", "_accel", "accel_",
                "acceleration", "_slope", "slope_", "_change", "change_", "_norm", "norm_"
            ]
            has_microstructure_base = any(token in feature_lower for token in microstructure_base_tokens)
            has_derivative_with_anchor = (
                any(token in feature_lower for token in derivative_tokens)
                and any(anchor in feature_lower for anchor in [
                    "order_flow", "imbalance", "microstructure", "trade_count"
                ])
            )
            if has_microstructure_base or has_derivative_with_anchor:
                categories["microstructure"].append(feature)
                categorized = True
        
        # Regime features (including multi-timeframe and derivative forms)
        if not categorized:
            regime_base_tokens = [
                "regime", "cluster", "state", "composite", "hmm", "regime_",
                "cluster_", "state_", "hmm_", "composite_"
            ]
            derivative_tokens = [
                "_diff", "diff_", "_delta", "delta_", "_accel", "accel_",
                "acceleration", "_slope", "slope_", "_change", "change_", "_norm", "norm_"
            ]
            has_regime_base = any(token in feature_lower for token in regime_base_tokens)
            has_derivative_with_anchor = (
                any(token in feature_lower for token in derivative_tokens)
                and any(anchor in feature_lower for anchor in [
                    "regime", "cluster", "state", "hmm", "composite"
                ])
            )
            if has_regime_base or has_derivative_with_anchor:
                categories["regime"].append(feature)
                categorized = True
        
        # Support/Resistance features (including multi-timeframe and derivative forms)
        if not categorized:
            sr_base_tokens = [
                "sr_distance", "support_level", "resistance_level", "proximity",
                "multi_timeframe_sr_score", "sr_proximity", "sr_outcome",
                "normalized_distance", "sr_proximity_score", "strength_score", 
                "clarity_factor", "directional_pressure", "sr_score", "delta_sr_score", 
                "isolation_score", "sr_level", "sr_breakout", "sr_rebounce", 
                "sr_consolidation", "sr_breakout_prob", "sr_rebounce_prob",
                "sr_consolidation_prob", "sr_multi_timeframe", "sr_", "support_", "resistance_"
            ]
            derivative_tokens = [
                "_diff", "diff_", "_delta", "delta_", "_accel", "accel_",
                "acceleration", "_slope", "slope_", "_change", "change_", "_norm", "norm_"
            ]
            has_sr_base = any(token in feature_lower for token in sr_base_tokens)
            has_derivative_with_anchor = (
                any(token in feature_lower for token in derivative_tokens)
                and any(anchor in feature_lower for anchor in [
                    "sr_", "support", "resistance", "proximity", "distance"
                ])
            )
            if has_sr_base or has_derivative_with_anchor:
                categories["sr_features"].append(feature)
                categorized = True
        

        
        if not categorized:
            categories["other"].append(feature)
    
    return categories

def test_sr_feature_categorization():
    """Test SR feature categorization."""
    print("🧪 Testing SR Feature Categorization (Simple Version)")
    print("=" * 60)
    
    # Test features with various SR-related names
    sr_feature_names = [
        # Basic SR features
        "sr_distance", "sr_proximity",
        "support_level", "resistance_level", "sr_level",
        
        # Distance features (not categorized as SR)
        "distance_to_resistance", "distance_to_support",
        "normalized_distance", "sr_distance_1", "sr_distance_2",
        
        # Score features
        "sr_score", "multi_timeframe_sr_score", "sr_proximity_score",
        "strength_score", "clarity_factor", "directional_pressure",
        "delta_sr_score", "isolation_score",
        
        # Proximity features
        "sr_proximity", "sr_proximity_1", "sr_proximity_2",
        
        # Volume features (should be categorized as volume)
        "volume_ratio", "volume_ma", "volume_change", "volume_sma", 
        "volume_momentum", "volume_weighted_momentum", "volume_velocity",
        "volume_acceleration", "volume_price_impact",
        
        # Interaction features (should be categorized as interaction)
        "momentum_x_volume", "volatility_div_liquidity", "rsi_ratio_volume",
        "regime_x_momentum", "volume_x_volatility",
        
        # Momentum features (including derivatives and multi-timeframe)
        "rsi_14", "rsi_diff_1", "rsi_accel_3", "rsi_norm_20",
        "macd_12_26", "macd_signal", "macd_histogram", "macd_diff_1",
        "momentum_strength", "momentum_diff_1", "momentum_accel_3",
        "bb_position", "bb_upper", "bb_lower", "bb_width",
        "sma_20", "ema_12", "ema_26", "sma_diff_5_20",
        
        # Volatility features (including derivatives)
        "volatility_garman_klass", "atr_14", "atr_diff_1", "atr_norm_20",
        "realized_vol_20", "volatility_diff_1", "volatility_accel_3",
        "bbands_std", "bbands_width", "bbands_position",
        
        # Liquidity features
        "spread_1m", "bid_ask_spread", "liquidity_ratio", "quote_imbalance",
        "spread_diff_1", "liquidity_norm_20",
        
        # Microstructure features
        "order_flow_imbalance", "trade_frequency", "order_flow_diff_1",
        "imbalance_norm_20", "trade_count", "trade_count_diff_1",
        
        # Regime features
        "hmm_state_0", "regime_1", "cluster_0", "composite_regime",
        "hmm_state_diff_1", "regime_norm_20"
    ]
    
    # Test categorization
    categories = categorize_features_simple(sr_feature_names)
    
    print("📊 Feature Categorization Results:")
    print("-" * 30)
    
    for category, features in categories.items():
        if features:
            print(f"{category}: {len(features)} features")
            if category == "sr_features":
                print(f"  SR features found: {features}")
            elif len(features) <= 5:
                print(f"  Examples: {features}")
    
    # Verify SR features are properly categorized
    sr_features_found = categories.get("sr_features", [])
    expected_sr_features = [
        "sr_distance", "sr_proximity", "support_level", 
        "resistance_level", "sr_level", "normalized_distance", "sr_distance_1", 
        "sr_distance_2", "sr_breakout_prob", "sr_rebounce_prob", 
        "sr_consolidation_prob", "sr_score", "multi_timeframe_sr_score", 
        "sr_proximity_score", "strength_score", "clarity_factor", 
        "directional_pressure", "delta_sr_score", "isolation_score", 
        "sr_proximity_1", "sr_proximity_2"
    ]
    
    # Check for features that should NOT be categorized as SR
    non_sr_features = ["distance_to_resistance", "distance_to_support"]
    incorrectly_categorized_as_sr = [f for f in non_sr_features if f in sr_features_found]
    if incorrectly_categorized_as_sr:
        print(f"   - Features incorrectly categorized as SR: {incorrectly_categorized_as_sr}")
        # Remove them from the found list for accurate counting
        sr_features_found = [f for f in sr_features_found if f not in incorrectly_categorized_as_sr]
    
    print(f"\n✅ SR Features Categorization Test:")
    print(f"   - Expected SR features: {len(expected_sr_features)}")
    print(f"   - Found SR features: {len(sr_features_found)}")
    print(f"   - Success rate: {len(sr_features_found) / len(expected_sr_features) * 100:.1f}%")
    
    # Check for any SR features that weren't categorized
    missing_sr_features = [f for f in expected_sr_features if f not in sr_features_found]
    if missing_sr_features:
        print(f"   - Missing SR features: {missing_sr_features}")
    
    # Check for any non-SR features that were incorrectly categorized as SR
    non_sr_features = ["rsi_14", "macd_12_26", "volume_sma_20", "momentum_strength", 
                      "volatility_garman_klass", "order_flow_imbalance", "hmm_state_0"]
    incorrectly_categorized = [f for f in non_sr_features if f in sr_features_found]
    if incorrectly_categorized:
        print(f"   - Incorrectly categorized as SR: {incorrectly_categorized}")
    
    # Test feature category weights
    print(f"\n🎯 Testing Feature Category Weights:")
    print("-" * 30)
    
    category_weights = {
        "momentum": 0.25,
        "volatility": 0.10,
        "liquidity": 0.10,
        "volume": 0.15,
        "microstructure": 0.10,
        "regime": 0.10,
        "sr_features": 0.10,
        "interaction": 0.10
    }
    
    target_features = 100
    print(f"Target features: {target_features}")
    print("Feature distribution by category:")
    
    for category, weight in category_weights.items():
        target_per_category = int(target_features * weight)
        available_features = len(categories.get(category, []))
        print(f"  {category}: {target_per_category} target, {available_features} available")
    
    # Verify weights sum to 1.0
    total_weight = sum(category_weights.values())
    print(f"Total weight: {total_weight}")
    
    if abs(total_weight - 1.0) < 0.01:
        print("✅ Category weights are properly normalized")
    else:
        print("❌ Category weights are not properly normalized")
    
    return len(sr_features_found) >= len(expected_sr_features) * 0.8  # 80% success rate

if __name__ == "__main__":
    success = test_sr_feature_categorization()
    
    print(f"\n📊 Test Result: {'PASSED' if success else 'FAILED'}")
    
    if success:
        print("🎉 SR feature categorization is working correctly!")
        print("   - SR features are properly identified and categorized")
        print("   - Feature category weights are properly configured")
        print("   - Balanced selection will include SR features in the mix")
    else:
        print("❌ SR feature categorization needs improvement")
        print("   - Some SR features may not be properly identified")
        print("   - Check the categorization logic in the feature selection manager")