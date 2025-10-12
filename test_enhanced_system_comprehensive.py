"""
Comprehensive Test Script for Enhanced Data-Driven Feature Selection and Interaction Generation

This script tests the complete system with comprehensive logging and error handling.
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Optional

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data(n_samples: int = 1000, n_features: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    """Create test data for the system."""
    print("🔧 Creating test data...")
    
    # Create synthetic OHLCV data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Generate price data
    close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    high_prices = close_prices + np.abs(np.random.randn(n_samples) * 0.5)
    low_prices = close_prices - np.abs(np.random.randn(n_samples) * 0.5)
    open_prices = close_prices + np.random.randn(n_samples) * 0.1
    volumes = np.random.randint(1000, 10000, n_samples)
    
    # Create base DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    # Add some basic features
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    data['volatility'] = data['returns'].rolling(20).std()
    data['sma_20'] = data['close'].rolling(20).mean()
    data['ema_12'] = data['close'].ewm(span=12).mean()
    data['rsi_14'] = calculate_rsi(data['close'], 14)
    data['atr_14'] = calculate_atr(data, 14)
    
    # Add more synthetic features
    for i in range(n_features - 8):  # We already have 8 features
        feature_name = f'feature_{i+1}'
        data[feature_name] = np.random.randn(n_samples) * np.random.uniform(0.1, 2.0)
    
    # Create target variable (simulate some relationship)
    target = (data['returns'].shift(-1) > 0).astype(int)  # Next period positive return
    
    print(f"✅ Created test data: {data.shape[0]} samples, {data.shape[1]} features")
    return data, target

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR indicator."""
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def test_data_driven_interaction_generator():
    """Test the DataDrivenInteractionGenerator."""
    print("\n" + "="*80)
    print("🧪 TESTING: DataDrivenInteractionGenerator")
    print("="*80)
    
    try:
        from src.feature_generation.utils.data_driven_interaction_generator import (
            DataDrivenInteractionGenerator, 
            EnhancedInteractionConfig
        )
        
        # Create test data
        data, targets = create_test_data(n_samples=500, n_features=15)
        
        # Create configuration
        config = EnhancedInteractionConfig(
            max_interactions=20,
            utility_threshold=0.05,
            enable_vectorbt=True,
            enable_parallel=True,
            memory_efficient=True
        )
        
        # Initialize generator
        print("🔧 Initializing DataDrivenInteractionGenerator...")
        generator = DataDrivenInteractionGenerator(config=config)
        print("✅ Generator initialized successfully")
        
        # Generate interactions
        print("⚡ Generating interactions...")
        interactions = generator.generate_interactions(data, targets)
        
        print(f"✅ Generated {len(interactions)} interactions")
        
        # Display results
        if interactions:
            print("\n📊 Top 5 interactions:")
            for i, interaction in enumerate(interactions[:5]):
                print(f"  {i+1}. {interaction.feature_name}")
                print(f"     Type: {interaction.interaction_type}")
                print(f"     Utility: {interaction.utility_score:.3f}")
                print(f"     Parents: {interaction.parent_features}")
                print()
        
        # Get performance stats
        stats = generator.get_performance_stats()
        print("📊 Performance Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: DataDrivenInteractionGenerator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_driven_feature_selector():
    """Test the DataDrivenFeatureSelector."""
    print("\n" + "="*80)
    print("🧪 TESTING: DataDrivenFeatureSelector")
    print("="*80)
    
    try:
        from src.feature_generation.utils.data_driven_feature_selector import (
            DataDrivenFeatureSelector,
            FeatureSelectionConfig
        )
        
        # Create test data
        data, targets = create_test_data(n_samples=500, n_features=15)
        
        # Create configuration
        config = FeatureSelectionConfig(
            target_feature_count=20,
            min_features_per_category=2,
            max_features_per_category=4,
            enable_vectorbt=True
        )
        
        # Initialize selector
        print("🔧 Initializing DataDrivenFeatureSelector...")
        selector = DataDrivenFeatureSelector(config=config)
        print("✅ Selector initialized successfully")
        
        # Select features
        print("🎯 Selecting features...")
        result = selector.select_features(data, targets)
        
        print(f"✅ Selected {len(result.selected_features)} features")
        print(f"📊 Categories: {result.category_distribution}")
        print(f"📊 Aspects: {result.aspect_distribution}")
        
        # Display selected features
        if result.selected_features:
            print("\n📊 Selected features:")
            for i, feature in enumerate(result.selected_features[:10]):  # Show first 10
                print(f"  {i+1}. {feature.feature_name}")
                print(f"     Category: {feature.category}")
                print(f"     Aspect: {feature.aspect_type}")
                print(f"     Score: {feature.score:.3f}")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: DataDrivenFeatureSelector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_data_driven_interaction_generator():
    """Test the EnhancedDataDrivenInteractionGenerator."""
    print("\n" + "="*80)
    print("🧪 TESTING: EnhancedDataDrivenInteractionGenerator")
    print("="*80)
    
    try:
        from src.feature_generation.utils.enhanced_data_driven_interaction_generator import (
            EnhancedDataDrivenInteractionGenerator,
            EnhancedDataDrivenConfig
        )
        
        # Create test data
        data, targets = create_test_data(n_samples=500, n_features=15)
        
        # Create configuration
        config = EnhancedDataDrivenConfig(
            target_feature_count=20,
            min_features_per_category=2,
            max_features_per_category=4,
            max_interactions=30,
            enable_vectorbt=True
        )
        
        # Initialize generator
        print("🔧 Initializing EnhancedDataDrivenInteractionGenerator...")
        generator = EnhancedDataDrivenInteractionGenerator(config=config)
        print("✅ Generator initialized successfully")
        
        # Generate interactions
        print("⚡ Generating enhanced interactions...")
        result = generator.generate_interactions(data, targets)
        
        print(f"✅ Generated {result.final_interaction_count} interactions from {result.final_feature_count} features")
        print(f"📊 Feature selection metrics: {result.feature_selection_metrics}")
        print(f"📊 Interaction metrics: {result.interaction_metrics}")
        
        # Display selected features
        if result.selected_features:
            print("\n📊 Selected features (first 10):")
            for i, feature in enumerate(result.selected_features[:10]):
                print(f"  {i+1}. {feature.feature_name}")
                print(f"     Category: {feature.category}")
                print(f"     Score: {feature.score:.3f}")
                print()
        
        # Display interactions
        if result.interactions:
            print("\n📊 Generated interactions (first 5):")
            for i, interaction in enumerate(result.interactions[:5]):
                print(f"  {i+1}. {interaction.feature_name}")
                print(f"     Type: {interaction.interaction_type}")
                print(f"     Utility: {interaction.utility_score:.3f}")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: EnhancedDataDrivenInteractionGenerator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n" + "="*80)
    print("🧪 TESTING: Error Handling")
    print("="*80)
    
    try:
        from src.feature_generation.utils.enhanced_data_driven_interaction_generator import (
            EnhancedDataDrivenInteractionGenerator,
            EnhancedDataDrivenConfig
        )
        
        # Test with None data
        print("🔧 Testing with None data...")
        generator = EnhancedDataDrivenInteractionGenerator()
        result = generator.generate_interactions(None)
        if result.metadata.get('error', False):
            print("✅ Correctly handled None data")
        else:
            print("❌ Failed to handle None data")
        
        # Test with empty DataFrame
        print("🔧 Testing with empty DataFrame...")
        empty_df = pd.DataFrame()
        result = generator.generate_interactions(empty_df)
        if result.metadata.get('error', False):
            print("✅ Correctly handled empty DataFrame")
        else:
            print("❌ Failed to handle empty DataFrame")
        
        # Test with invalid data type
        print("🔧 Testing with invalid data type...")
        result = generator.generate_interactions("invalid_data")
        if result.metadata.get('error', False):
            print("✅ Correctly handled invalid data type")
        else:
            print("❌ Failed to handle invalid data type")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 COMPREHENSIVE TEST SUITE FOR ENHANCED DATA-DRIVEN SYSTEM")
    print("="*80)
    
    tests = [
        ("DataDrivenInteractionGenerator", test_data_driven_interaction_generator),
        ("DataDrivenFeatureSelector", test_data_driven_feature_selector),
        ("EnhancedDataDrivenInteractionGenerator", test_enhanced_data_driven_interaction_generator),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running test: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ CRITICAL ERROR in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<40}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is fully wired and working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)