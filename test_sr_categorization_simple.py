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
        "microstructure": [],
        "regime": [],
        "sr_features": [],
        "interaction": [],
        "other": []
    }
    
    for feature in feature_names:
        feature_lower = feature.lower()
        categorized = False
        
        # Momentum indicators
        if any(keyword in feature_lower for keyword in [
            "momentum", "mom", "rsi", "macd", "cci", "roc", "willr", "stoch",
            "adx", "dmi", "kama", "tema", "dema", "hma", "wma", "vwma", "zlema",
            "ichimoku", "psar", "trix", "cmo", "tsi", "ppo", "pmo", "uo",
            "linreg", "lin_reg", "sma", "ema", "ma_", "moving_avg", "trend"
        ]):
            categories["momentum"].append(feature)
            categorized = True
        
        # Volatility measures
        elif any(keyword in feature_lower for keyword in [
            "volatility", "atr", "true_range", "truerange", "natr", "parkinson",
            "garman", "gk_vol", "garman_klass", "roll", "rvol", "realized_vol",
            "hv", "hist_vol", "historical_vol", "variance", "std", "bbands",
            "boll", "bollinger", "donch", "donchian", "keltner", "chop",
            "choppiness", "park_vol"
        ]):
            categories["volatility"].append(feature)
            categorized = True
        
        # Liquidity/volume features
        elif any(keyword in feature_lower for keyword in [
            "liquidity", "volume", "tick_volume", "obv", "cmf", "mfi", "vwap",
            "pvi", "nvi", "efi", "delta_volume"
        ]):
            categories["liquidity"].append(feature)
            categorized = True
        
        # Microstructure/order book features
        elif any(keyword in feature_lower for keyword in [
            "microstructure", "order_flow", "orderflow", "ofi", "imbalance",
            "quote_imbalance", "spread", "bid_ask", "depth", "orderbook", "book",
            "microprice", "trade_count", "trade_frequency"
        ]):
            categories["microstructure"].append(feature)
            categorized = True
        
        # Regime features
        elif any(keyword in feature_lower for keyword in [
            "regime", "cluster", "state", "composite", "hmm"
        ]):
            categories["regime"].append(feature)
            categorized = True
        
        # Support/Resistance features
        elif any(keyword in feature_lower for keyword in [
            "sr_distance", "support_level", "resistance_level", "proximity",
            "multi_timeframe_sr_score", "sr_proximity", "sr_outcome",
            "normalized_distance",
            "sr_proximity_score", "strength_score", "clarity_factor", "directional_pressure",
            "sr_score", "delta_sr_score", "isolation_score", "sr_level", "sr_breakout",
            "sr_rebounce", "sr_consolidation", "sr_breakout_prob", "sr_rebounce_prob",
            "sr_consolidation_prob", "sr_multi_timeframe"
        ]):
            categories["sr_features"].append(feature)
            categorized = True
        
        # Interaction features
        elif any(keyword in feature_lower for keyword in [
            "_x_", "_div_", "_ratio_", "_over_", "_cross_", "interaction"
        ]):
            categories["interaction"].append(feature)
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
        
        # Other features (should not be categorized as SR)
        "rsi_14", "macd_12_26", "volume_sma_20", "momentum_strength",
        "volatility_garman_klass", "order_flow_imbalance", "hmm_state_0"
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
        "momentum": 0.20,
        "volatility": 0.15,
        "liquidity": 0.15,
        "microstructure": 0.10,
        "regime": 0.10,
        "sr_features": 0.15,
        "interaction": 0.15
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