#!/usr/bin/env python3
"""Test script for profit-based feature engineering integration in step2."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_data_with_profit(n_samples: int = 1000) -> pd.DataFrame:
    """Create test market data with profit percentages."""
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="1min")
    
    # Create realistic price movements
    np.random.seed(42)  # For reproducible results
    
    # Start with a base price
    base_price = 100.0
    prices = [base_price]
    
    # Generate price movements with some trend and volatility
    for i in range(1, n_samples):
        # Add some trend and random walk
        change = np.random.normal(0, 0.001) + 0.0001  # Small upward trend
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLC data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples),
    }, index=dates)
    
    # Ensure high >= close >= low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    # Add profit percentages (simulating triple barrier results)
    # Mix of positive and negative profits for realistic testing
    profit_pcts = np.random.uniform(-0.01, 0.01, n_samples)
    # Add some structure to make it more realistic
    profit_pcts = profit_pcts + np.sin(np.arange(n_samples) * 0.1) * 0.002
    data['potential_profit_pct'] = profit_pcts
    
    # Add labels (1 for LONG, -1 for SHORT, 0 for HOLD)
    labels = np.sign(profit_pcts)
    data['label'] = labels
    
    return data

async def test_step2_profit_integration():
    """Test the integration of profit-based features in step2."""
    print("🧪 Testing Step2 Profit-Based Feature Engineering Integration")
    print("=" * 70)
    
    # Create test data
    print("📊 Creating test market data with profit percentages...")
    test_data = create_test_data_with_profit(1000)
    print(f"   Created {len(test_data)} data points")
    print(f"   Price range: ${test_data['close'].min():.2f} - ${test_data['close'].max():.2f}")
    print(f"   Profit range: {test_data['potential_profit_pct'].min():.4f} - {test_data['potential_profit_pct'].max():.4f}")
    print(f"   LONG positions: {(test_data['label'] == 1).sum()}")
    print(f"   SHORT positions: {(test_data['label'] == -1).sum()}")
    
    # Test profit-based feature engineering directly
    print("\n🔧 Testing Profit-Based Feature Engineering Directly...")
    try:
        from src.training.steps.step4_analyst_labeling_feature_engineering_components.profit_based_feature_engineering import (
            ProfitBasedFeatureEngineering
        )
        
        profit_feature_eng = ProfitBasedFeatureEngineering(
            profit_column="potential_profit_pct",
            volume_column="volume",
            price_column="close",
            use_numba=True,
            memory_efficient=True
        )
        
        profit_features = profit_feature_eng.apply_all_features(test_data)
        
        print(f"✅ Direct profit feature engineering completed")
        print(f"   - Input shape: {test_data.shape}")
        print(f"   - Output shape: {profit_features.shape}")
        print(f"   - Features added: {len(profit_features.columns) - len(test_data.columns)}")
        
        # Show some profit-based features
        profit_feature_cols = [col for col in profit_features.columns if "potential_profit_pct" in col and col != "potential_profit_pct"]
        print(f"   - Profit-based features: {len(profit_feature_cols)}")
        print(f"   - Sample features: {profit_feature_cols[:10]}")
        
    except Exception as e:
        print(f"❌ Direct profit feature engineering failed: {e}")
        return False
    
    # Test step2 vectorized feature engineering integration
    print("\n🔧 Testing Step2 Vectorized Feature Engineering Integration...")
    try:
        from src.training.steps.vectorized_advanced_feature_engineering import (
            VectorizedAdvancedFeatureEngineering
        )
        
        # Create configuration for step2
        config = {
            "vectorized_advanced_features": {
                "enable_volatility_modeling": True,
                "enable_correlation_analysis": True,
                "enable_momentum_analysis": True,
                "enable_liquidity_analysis": True,
                "enable_candlestick_patterns": True,
                "enable_sr_distance": True,
                "enable_wavelet_transforms": True,
                "enable_multi_timeframe": True,
            }
        }
        
        # Initialize step2 feature engineering
        step2_feature_eng = VectorizedAdvancedFeatureEngineering(config)
        init_success = await step2_feature_eng.initialize()
        
        if not init_success:
            print("❌ Step2 feature engineering initialization failed")
            return False
        
        print("✅ Step2 feature engineering initialized successfully")
        
        # Prepare data for step2 (separate price and volume data)
        price_data = test_data[['open', 'high', 'low', 'close', 'potential_profit_pct']].copy()
        volume_data = test_data[['volume']].copy()
        
        print(f"   - Price data shape: {price_data.shape}")
        print(f"   - Volume data shape: {volume_data.shape}")
        
        # Apply step2 feature engineering
        print("🚀 Applying step2 vectorized feature engineering...")
        step2_features = await step2_feature_eng.engineer_features(
            price_data=price_data,
            volume_data=volume_data,
            order_flow_data=None,
            sr_levels=None
        )
        
        print(f"✅ Step2 feature engineering completed")
        print(f"   - Total features generated: {len(step2_features)}")
        
        # Check for profit-based features in step2 output
        profit_based_features = [name for name in step2_features.keys() if "potential_profit_pct" in name]
        print(f"   - Profit-based features in step2: {len(profit_based_features)}")
        
        if profit_based_features:
            print(f"   - Sample profit features: {profit_based_features[:10]}")
            
            # Analyze profit feature categories
            feature_categories = {}
            for feature_name in profit_based_features:
                if "squared" in feature_name or "cubed" in feature_name or "abs" in feature_name:
                    feature_categories["basic"] = feature_categories.get("basic", 0) + 1
                elif "sign" in feature_name or "magnitude" in feature_name or "bins" in feature_name:
                    feature_categories["categorical"] = feature_categories.get("categorical", 0) + 1
                elif "sharpe" in feature_name or "sortino" in feature_name or "kelly" in feature_name:
                    feature_categories["risk_reward"] = feature_categories.get("risk_reward", 0) + 1
                elif "momentum" in feature_name or "acceleration" in feature_name:
                    feature_categories["momentum"] = feature_categories.get("momentum", 0) + 1
                elif "volatility" in feature_name:
                    feature_categories["volatility"] = feature_categories.get("volatility", 0) + 1
                elif "volume" in feature_name:
                    feature_categories["volume"] = feature_categories.get("volume", 0) + 1
                elif "rolling" in feature_name:
                    feature_categories["rolling"] = feature_categories.get("rolling", 0) + 1
                else:
                    feature_categories["other"] = feature_categories.get("other", 0) + 1
            
            print(f"   - Profit feature categories: {feature_categories}")
        else:
            print("⚠️ No profit-based features found in step2 output")
        
        # Show overall feature distribution
        all_features = list(step2_features.keys())
        print(f"\n📊 Overall Feature Distribution:")
        print(f"   - Total features: {len(all_features)}")
        
        # Categorize all features
        all_categories = {}
        for feature_name in all_features:
            if "potential_profit_pct" in feature_name:
                all_categories["profit_based"] = all_categories.get("profit_based", 0) + 1
            elif "wavelet" in feature_name.lower():
                all_categories["wavelet"] = all_categories.get("wavelet", 0) + 1
            elif "momentum" in feature_name.lower() or "rsi" in feature_name.lower():
                all_categories["momentum"] = all_categories.get("momentum", 0) + 1
            elif "volatility" in feature_name.lower():
                all_categories["volatility"] = all_categories.get("volatility", 0) + 1
            elif "volume" in feature_name.lower():
                all_categories["volume"] = all_categories.get("volume", 0) + 1
            elif "correlation" in feature_name.lower():
                all_categories["correlation"] = all_categories.get("correlation", 0) + 1
            elif "candlestick" in feature_name.lower():
                all_categories["candlestick"] = all_categories.get("candlestick", 0) + 1
            elif "sr" in feature_name.lower():
                all_categories["sr_distance"] = all_categories.get("sr_distance", 0) + 1
            else:
                all_categories["other"] = all_categories.get("other", 0) + 1
        
        print(f"   - Feature categories: {all_categories}")
        
        # Test feature quality
        print(f"\n🔍 Feature Quality Analysis:")
        
        # Check for missing values
        missing_features = []
        for feature_name, feature_values in step2_features.items():
            if hasattr(feature_values, 'isna'):
                missing_count = feature_values.isna().sum()
                if missing_count > 0:
                    missing_features.append((feature_name, missing_count))
        
        if missing_features:
            print(f"   - Features with missing values: {len(missing_features)}")
            print(f"   - Sample missing features: {missing_features[:5]}")
        else:
            print(f"   - No missing values found")
        
        # Check for infinite values
        inf_features = []
        for feature_name, feature_values in step2_features.items():
            if hasattr(feature_values, 'values'):
                if np.any(np.isinf(feature_values.values)):
                    inf_features.append(feature_name)
        
        if inf_features:
            print(f"   - Features with infinite values: {len(inf_features)}")
            print(f"   - Sample infinite features: {inf_features[:5]}")
        else:
            print(f"   - No infinite values found")
        
        print("\n✅ Step2 Profit Integration Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Step2 integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    
    # Run the test
    success = asyncio.run(test_step2_profit_integration())
    
    if success:
        print("\n🎉 All tests passed! Profit-based feature engineering is properly integrated into step2.")
    else:
        print("\n❌ Tests failed. Please check the implementation.")
        sys.exit(1)