#!/usr/bin/env python3
"""Test script for exit strategy feature engineering system."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_data_with_positions(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic test data with positions and profit information."""
    print(f"📊 Creating test data with {n_samples} samples...")
    
    # Generate timestamps
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="1min")
    
    # Generate price data with some trends
    np.random.seed(42)  # For reproducible results
    
    # Create price series with trends and reversals
    price_trend = np.cumsum(np.random.normal(0, 0.001, n_samples))
    price_noise = np.random.normal(0, 0.002, n_samples)
    base_price = 100 + price_trend + price_noise
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': base_price + np.random.normal(0, 0.001, n_samples),
        'high': base_price + np.abs(np.random.normal(0, 0.002, n_samples)),
        'low': base_price - np.abs(np.random.normal(0, 0.002, n_samples)),
        'close': base_price,
        'volume': np.random.uniform(1000, 10000, n_samples),
    }, index=dates)
    
    # Generate position data (1=LONG, -1=SHORT, 0=no position)
    # Create some realistic position patterns
    positions = np.zeros(n_samples)
    current_position = 0
    position_duration = 0
    
    for i in range(n_samples):
        # Randomly change positions
        if np.random.random() < 0.01:  # 1% chance to change position
            if current_position == 0:
                current_position = np.random.choice([1, -1])
            else:
                current_position = 0
            position_duration = 0
        else:
            position_duration += 1
        
        positions[i] = current_position
    
    data['position'] = positions
    
    # Generate profit percentage data
    profit_pcts = np.zeros(n_samples)
    for i in range(1, n_samples):
        if positions[i-1] != 0:  # If we had a position
            # Calculate profit based on price movement
            price_change = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
            profit_pcts[i] = price_change * positions[i-1]  # Profit depends on position direction
    
    data['potential_profit_pct'] = profit_pcts
    
    print(f"✅ Test data created:")
    print(f"   - Shape: {data.shape}")
    print(f"   - Position distribution: {pd.Series(positions).value_counts().to_dict()}")
    print(f"   - Profit range: {data['potential_profit_pct'].min():.4f} to {data['potential_profit_pct'].max():.4f}")
    
    return data

def test_exit_strategy_feature_engineering():
    """Test the exit strategy feature engineering system."""
    print("\n🔧 Testing Exit Strategy Feature Engineering...")
    
    try:
        from src.training.steps.exit_strategy_feature_engineering import ExitStrategyFeatureEngineering
        
        # Create test data
        test_data = create_test_data_with_positions(1000)
        
        # Initialize exit strategy feature engineering
        exit_strategy = ExitStrategyFeatureEngineering(
            price_column="close",
            volume_column="volume",
            position_column="position",
            profit_column="potential_profit_pct",
            confidence_threshold=0.6,
            use_numba=False,  # Use Python for testing
            memory_efficient=True
        )
        
        # Apply all features
        result = exit_strategy.apply_all_features(test_data)
        
        # Check that features were added
        original_columns = set(test_data.columns)
        result_columns = set(result.columns)
        new_features = result_columns - original_columns
        
        print(f"✅ Exit strategy features applied:")
        print(f"   - Original columns: {len(original_columns)}")
        print(f"   - Result columns: {len(result_columns)}")
        print(f"   - New features: {len(new_features)}")
        
        # Check for specific feature categories
        feature_categories = {
            "momentum_reversal": [col for col in new_features if "momentum" in col],
            "volatility_reversal": [col for col in new_features if "volatility" in col],
            "volume_reversal": [col for col in new_features if "volume" in col and "reversal" in col],
            "support_resistance": [col for col in new_features if "support" in col or "resistance" in col or "sr_" in col],
            "trend_strength": [col for col in new_features if "trend" in col],
            "profit_decay": [col for col in new_features if "profit" in col and "decay" in col],
            "time_decay": [col for col in new_features if "time" in col or "duration" in col],
            "market_regime": [col for col in new_features if "regime" in col],
        }
        
        for category, features in feature_categories.items():
            print(f"   - {category}: {len(features)} features")
            if features:
                print(f"     Sample: {features[:3]}")
        
        # Test exit confidence calculation
        exit_confidence = exit_strategy.calculate_exit_confidence(result)
        print(f"✅ Exit confidence calculated:")
        print(f"   - Confidence range: {exit_confidence.min():.4f} to {exit_confidence.max():.4f}")
        print(f"   - Mean confidence: {exit_confidence.mean():.4f}")
        print(f"   - Non-zero confidence points: {(exit_confidence > 0).sum()}")
        
        # Test exit recommendations
        recommendations = exit_strategy.get_exit_recommendations(result, confidence_threshold=0.6)
        print(f"✅ Exit recommendations generated:")
        print(f"   - Total recommendations: {len(recommendations)}")
        print(f"   - Exit signals: {recommendations['should_exit'].sum()}")
        print(f"   - Exit rate: {recommendations['should_exit'].mean():.2%}")
        
        # Show exit reasons
        if recommendations['should_exit'].sum() > 0:
            exit_reasons = recommendations[recommendations['should_exit']]['exit_reason'].value_counts()
            print(f"   - Top exit reasons: {exit_reasons.head().to_dict()}")
        
        # Test feature summary
        summary = exit_strategy.get_feature_summary(result)
        print(f"✅ Feature summary generated:")
        print(f"   - Total exit features: {summary['total_features']}")
        print(f"   - Performance metrics: {summary['performance_metrics']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Exit strategy feature engineering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_exit_strategy_integration():
    """Test integration with existing profit-based features."""
    print("\n🔧 Testing Exit Strategy Integration...")
    
    try:
        from src.training.steps.exit_strategy_feature_engineering import ExitStrategyFeatureEngineering
        from src.training.steps.step4_analyst_labeling_feature_engineering_components.profit_based_feature_engineering import (
            ProfitBasedFeatureEngineering
        )
        
        # Create test data
        test_data = create_test_data_with_positions(500)
        
        # Apply profit-based features first
        profit_feature_engineer = ProfitBasedFeatureEngineering(
            profit_column="potential_profit_pct",
            volume_column="volume",
            price_column="close",
            use_numba=False,
            memory_efficient=True
        )
        
        data_with_profit_features = profit_feature_engineer.apply_all_features(test_data)
        print(f"✅ Applied profit-based features: {len(data_with_profit_features.columns)} columns")
        
        # Apply exit strategy features
        exit_strategy = ExitStrategyFeatureEngineering(
            price_column="close",
            volume_column="volume",
            position_column="position",
            profit_column="potential_profit_pct",
            confidence_threshold=0.6,
            use_numba=False,
            memory_efficient=True
        )
        
        final_data = exit_strategy.apply_all_features(data_with_profit_features)
        print(f"✅ Applied exit strategy features: {len(final_data.columns)} total columns")
        
        # Check for both feature types
        profit_features = [col for col in final_data.columns if "potential_profit_pct" in col and col != "potential_profit_pct"]
        exit_features = [col for col in final_data.columns if any(x in col for x in [
            "reversal_prob", "momentum", "volatility", "volume", "support", 
            "resistance", "trend", "profit_decay", "time_decay", "market_regime"
        ])]
        
        print(f"✅ Integration test results:")
        print(f"   - Profit-based features: {len(profit_features)}")
        print(f"   - Exit strategy features: {len(exit_features)}")
        print(f"   - Total features: {len(final_data.columns)}")
        
        # Test exit confidence with combined features
        exit_confidence = exit_strategy.calculate_exit_confidence(final_data)
        recommendations = exit_strategy.get_exit_recommendations(final_data, confidence_threshold=0.6)
        
        print(f"   - Exit signals: {recommendations['should_exit'].sum()} out of {len(recommendations)}")
        print(f"   - Exit rate: {recommendations['should_exit'].mean():.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Exit strategy integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_exit_strategy_performance():
    """Test performance with different data sizes."""
    print("\n🔧 Testing Exit Strategy Performance...")
    
    try:
        from src.training.steps.exit_strategy_feature_engineering import ExitStrategyFeatureEngineering
        import time
        
        # Test different data sizes
        sizes = [100, 500, 1000, 2000]
        
        for size in sizes:
            print(f"   Testing with {size} samples...")
            
            # Create test data
            test_data = create_test_data_with_positions(size)
            
            # Initialize exit strategy
            exit_strategy = ExitStrategyFeatureEngineering(
                use_numba=False,  # Use Python for consistent testing
                memory_efficient=True
            )
            
            # Time the feature application
            start_time = time.time()
            result = exit_strategy.apply_all_features(test_data)
            processing_time = time.time() - start_time
            
            # Calculate exit confidence
            exit_confidence = exit_strategy.calculate_exit_confidence(result)
            recommendations = exit_strategy.get_exit_recommendations(result)
            
            print(f"     ✅ Size {size}: {processing_time:.3f}s, {len(result.columns)} features, "
                  f"{recommendations['should_exit'].sum()} exit signals")
        
        return True
        
    except Exception as e:
        print(f"❌ Exit strategy performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all exit strategy tests."""
    print("🚀 Starting Exit Strategy Feature Engineering Tests")
    print("=" * 60)
    
    tests = [
        ("Exit Strategy Feature Engineering", test_exit_strategy_feature_engineering),
        ("Exit Strategy Integration", test_exit_strategy_integration),
        ("Exit Strategy Performance", test_exit_strategy_performance),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All exit strategy tests passed!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)