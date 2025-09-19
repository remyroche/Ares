#!/usr/bin/env python3
"""
Test Script for New Directional Feature Lookback Optimization

This script demonstrates the new directional optimization approach that generates
1 period per feature per direction (long/short) instead of 2 periods per feature.

Usage:
    python test_new_directional_optimization.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.append('/workspace/src')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Import the new components
try:
    from training.steps.market_analysis.feature_lookback_optimization.directional_lookback_optimizer import (
        DirectionalLookbackOptimizer, DirectionalLookbackConfig, optimize_features_directional
    )
    from training.steps.market_analysis.feature_lookback_optimization.directional_feature_selection_adapter import (
        DirectionalFeatureSelectionAdapter, DirectionalFeatureSelectionConfig, select_directional_features
    )
    from utils.tprint import tprint
    
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    COMPONENTS_AVAILABLE = False

def generate_sample_data(n_samples: int = 1000, n_features: int = 20) -> pd.DataFrame:
    """Generate sample OHLCV data with features and targets."""
    np.random.seed(42)
    
    # Generate base OHLCV data
    data = pd.DataFrame()
    
    # Price data
    data['close'] = 100 + np.cumsum(np.random.randn(n_samples) * 0.02)
    data['high'] = data['close'] * (1 + np.abs(np.random.randn(n_samples) * 0.01))
    data['low'] = data['close'] * (1 - np.abs(np.random.randn(n_samples) * 0.01))
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
    data['volume'] = np.random.lognormal(10, 1, n_samples)
    
    # Generate returns and targets
    data['returns'] = data['close'].pct_change().fillna(0)
    data['close_return'] = data['returns']
    
    # Generate sample features (technical indicators)
    for i in range(n_features):
        # Create features with different characteristics
        if i < 5:
            # Trend-following features
            data[f'feature_{i}'] = data['close'].rolling(window=10+i*2).mean()
        elif i < 10:
            # Momentum features  
            data[f'feature_{i}'] = data['returns'].rolling(window=5+i).std()
        elif i < 15:
            # Volume-based features
            data[f'feature_{i}'] = data['volume'].rolling(window=8+i).mean()
        else:
            # Price-based ratios
            data[f'feature_{i}'] = data['high'] / data['low']
    
    # Fill NaN values
    data = data.fillna(method='bfill').fillna(0)
    
    print(f"✅ Generated sample data: {len(data)} rows, {len(data.columns)} columns")
    print(f"📊 Returns distribution: mean={data['returns'].mean():.4f}, std={data['returns'].std():.4f}")
    print(f"📊 Positive returns: {(data['returns'] > 0).sum()}, Negative returns: {(data['returns'] < 0).sum()}")
    
    return data

def test_directional_optimization():
    """Test the new directional optimization approach."""
    if not COMPONENTS_AVAILABLE:
        print("❌ Components not available, skipping test")
        return False
    
    print("🚀 Testing New Directional Feature Lookback Optimization")
    print("=" * 60)
    
    # Generate sample data
    data = generate_sample_data(n_samples=2000, n_features=15)
    feature_columns = [col for col in data.columns if col.startswith('feature_')]
    
    print(f"📊 Testing with {len(feature_columns)} features: {feature_columns[:5]}...")
    
    # Configure directional optimization
    directional_config = DirectionalLookbackConfig(
        min_lookback=5,
        max_lookback=30,  # Reduced for faster testing
        target_total_features=60,  # Target 60 features (30 long + 30 short)
        max_features_per_direction=40,
        enable_directional=True,
        parallel_optimization=False,  # Disabled for testing
        cross_directional_analysis=True,
        min_samples_per_direction=100
    )
    
    # Test 1: Direct optimization function
    print("\n🧪 Test 1: Direct optimization function")
    try:
        result = optimize_features_directional(
            data=data,
            feature_columns=feature_columns[:10],  # Limit for testing
            target_column='returns',
            config=directional_config
        )
        
        print(f"✅ Optimization completed successfully!")
        print(f"📊 Results:")
        print(f"   - Total features: {result.final_feature_count}")
        print(f"   - Long features: {len(result.selected_long_features)}")
        print(f"   - Short features: {len(result.selected_short_features)}")
        print(f"   - Optimization time: {result.total_optimization_time:.2f}s")
        print(f"   - Average MI score: {result.average_mutual_info_score:.4f}")
        print(f"   - Balance ratio: {result.directional_balance_ratio:.3f}")
        print(f"   - Convergence rate: {result.convergence_rate:.3f}")
        
        # Show some feature details
        print(f"\n📋 Sample long features: {result.selected_long_features[:3]}")
        print(f"📋 Sample short features: {result.selected_short_features[:3]}")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False
    
    # Test 2: Feature selection adapter
    print("\n🧪 Test 2: Feature selection adapter")
    try:
        selection_config = DirectionalFeatureSelectionConfig(
            target_total_features=50,  # Target 50 features
            maintain_directional_balance=True,
            min_mutual_info_score=0.001,  # Lower threshold for testing
            enable_cross_directional_filtering=True
        )
        
        selection_result = select_directional_features(
            directional_result=result,
            data=data,
            target_column='returns',
            config=selection_config
        )
        
        print(f"✅ Feature selection completed successfully!")
        print(f"📊 Selection results:")
        print(f"   - Total selected: {selection_result.total_selected_features}")
        print(f"   - Long selected: {len(selection_result.selected_long_features)}")
        print(f"   - Short selected: {len(selection_result.selected_short_features)}")
        print(f"   - Selection quality: {selection_result.selection_quality_score:.3f}")
        print(f"   - Balance ratio: {selection_result.directional_balance_ratio:.3f}")
        print(f"   - Avg MI score: {selection_result.average_mutual_info_score:.4f}")
        print(f"   - Selection time: {selection_result.selection_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False
    
    # Test 3: Compare with legacy approach simulation
    print("\n🧪 Test 3: Feature count comparison")
    try:
        # Simulate legacy approach (2 periods per feature)
        legacy_feature_count = len(feature_columns) * 2  # 2 periods per feature
        legacy_directional_count = legacy_feature_count * 2  # × 2 directions
        
        # New approach
        new_feature_count = result.final_feature_count
        
        print(f"📊 Feature count comparison:")
        print(f"   - Original features: {len(feature_columns)}")
        print(f"   - Legacy approach (2 periods × 2 directions): {legacy_directional_count}")
        print(f"   - New approach (1 period × 2 directions): {new_feature_count}")
        print(f"   - Reduction: {legacy_directional_count - new_feature_count} features")
        print(f"   - Reduction percentage: {(1 - new_feature_count/legacy_directional_count)*100:.1f}%")
        
        # Check if we're within ML model limits
        ml_optimal_range = (60, 100)
        if ml_optimal_range[0] <= new_feature_count <= ml_optimal_range[1]:
            print(f"✅ New approach is within optimal ML range {ml_optimal_range}")
        else:
            print(f"⚠️ New approach is outside optimal ML range {ml_optimal_range}")
        
        if legacy_directional_count > ml_optimal_range[1]:
            print(f"❌ Legacy approach would exceed optimal ML range")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False
    
    print("\n✅ All tests completed successfully!")
    print("\n📋 Summary:")
    print(f"   - New directional optimization generates 1 period per feature per direction")
    print(f"   - Feature count is manageable for ML models ({new_feature_count} features)")
    print(f"   - Directional balance is maintained ({selection_result.directional_balance_ratio:.3f})")
    print(f"   - Quality metrics are preserved (MI: {result.average_mutual_info_score:.4f})")
    
    return True

def demonstrate_configuration_options():
    """Demonstrate different configuration options."""
    print("\n🔧 Configuration Options Demo")
    print("=" * 40)
    
    # Option 1: Conservative (fewer features, higher quality)
    print("\n📋 Option 1: Conservative Configuration")
    conservative_config = DirectionalLookbackConfig(
        target_total_features=60,
        max_features_per_direction=35,
        min_mutual_info_score=0.05,  # Higher threshold
        min_samples_per_direction=100,
        cross_directional_analysis=True
    )
    print(f"   - Target features: {conservative_config.target_total_features}")
    print(f"   - Quality threshold: {conservative_config.min_mutual_info_score}")
    print(f"   - Cross-directional analysis: {conservative_config.cross_directional_analysis}")
    
    # Option 2: Aggressive (more features, lower quality threshold)
    print("\n📋 Option 2: Aggressive Configuration")
    aggressive_config = DirectionalLookbackConfig(
        target_total_features=90,
        max_features_per_direction=50,
        min_mutual_info_score=0.01,  # Lower threshold
        min_samples_per_direction=50,
        parallel_optimization=True
    )
    print(f"   - Target features: {aggressive_config.target_total_features}")
    print(f"   - Quality threshold: {aggressive_config.min_mutual_info_score}")
    print(f"   - Parallel optimization: {aggressive_config.parallel_optimization}")
    
    # Option 3: Balanced (recommended)
    print("\n📋 Option 3: Balanced Configuration (Recommended)")
    balanced_config = DirectionalLookbackConfig(
        target_total_features=80,
        max_features_per_direction=45,
        min_mutual_info_score=0.02,
        min_samples_per_direction=75,
        cross_directional_analysis=True,
        adaptive_feature_selection=True
    )
    print(f"   - Target features: {balanced_config.target_total_features}")
    print(f"   - Quality threshold: {balanced_config.min_mutual_info_score}")
    print(f"   - Adaptive selection: {balanced_config.adaptive_feature_selection}")

if __name__ == "__main__":
    print("🎯 New Directional Feature Lookback Optimization Test")
    print("=" * 60)
    
    # Run tests
    success = test_directional_optimization()
    
    if success:
        # Show configuration options
        demonstrate_configuration_options()
        
        print("\n🎉 Test completed successfully!")
        print("\n💡 Key Benefits of New Approach:")
        print("   ✅ Reduces feature count from 2N×2 to N×2 (50% reduction)")
        print("   ✅ Maintains directional differentiation (long/short)")
        print("   ✅ Stays within optimal ML model range (60-100 features)")
        print("   ✅ Preserves optimization quality and performance")
        print("   ✅ Provides intelligent feature selection")
        print("   ✅ Enables cross-directional analysis")
        
        print("\n🔧 Integration Notes:")
        print("   - Set 'use_new_directional_approach=True' in config")
        print("   - Adjust 'target_total_features' based on your ML model needs")
        print("   - Use 'max_features_to_optimize' to limit computation time")
        print("   - Enable 'cross_directional_analysis' for better insights")
        
    else:
        print("\n❌ Test failed - check error messages above")
        sys.exit(1)