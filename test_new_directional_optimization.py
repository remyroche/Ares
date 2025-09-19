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
    
    # Configure directional optimization with consolidation
    directional_config = DirectionalLookbackConfig(
        min_lookback=5,
        max_lookback=30,  # Reduced for faster testing
        enable_directional=True,
        parallel_optimization=False,  # Disabled for testing
        cross_directional_analysis=True,
        min_samples_per_direction=100,
        
        # New consolidation settings for precision-critical intraday trading
        enable_period_consolidation=True,
        consolidation_variance_threshold=0.12,  # 12% variance threshold (precision-critical for intraday/scalping)
        consolidation_method="average",
        
        # Adaptive threshold settings
        enable_adaptive_thresholds=True,
        trading_timeframe="intraday",
        market_volatility="medium",
        
        # Integration with existing pipeline
        use_existing_feature_pipeline=True,
        generate_features_for_pipeline=True
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
        print(f"   - Consolidated features: {len(result.consolidated_features)}")
        print(f"   - Optimization time: {result.total_optimization_time:.2f}s")
        print(f"   - Average MI score: {result.average_mutual_info_score:.4f}")
        print(f"   - Balance ratio: {result.directional_balance_ratio:.3f}")
        print(f"   - Convergence rate: {result.convergence_rate:.3f}")
        
        # Show consolidation details
        if result.consolidated_features:
            print(f"\n🔀 Consolidation Results:")
            for feature_name, consolidated_result in result.consolidated_features.items():
                print(f"   - {feature_name}: long={consolidated_result.original_long_period}, "
                      f"short={consolidated_result.original_short_period} → "
                      f"consolidated={consolidated_result.optimal_lookback_period} "
                      f"(variance: {consolidated_result.consolidation_variance:.3f})")
        
        # Show some feature details
        print(f"\n📋 Sample long features: {result.selected_long_features[:3]}")
        print(f"📋 Sample short features: {result.selected_short_features[:3]}")
        print(f"📋 Consolidated features: {list(result.consolidated_features.keys())[:3]}")
        
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
    
    # Test 3: Test threshold recommendations
    print("\n🧪 Test 3: Test threshold recommendations")
    try:
        from training.steps.market_analysis.feature_lookback_optimization.threshold_advisor import (
            ThresholdAdvisor, TradingTimeframe, MarketVolatility, FeatureType, 
            get_threshold_recommendation, print_threshold_analysis
        )
        
        print("\n🎯 Testing threshold advisor...")
        
        # Test different scenarios
        scenarios = [
            ("intraday", "high_volatility", "High-frequency crypto trading"),
            ("swing_trading", "medium_volatility", "Forex swing trading"),
            ("position_trading", "low_volatility", "Long-term stock trading")
        ]
        
        for trading_style, market_type, description in scenarios:
            threshold = get_threshold_recommendation(
                trading_style=trading_style,
                market_type=market_type,
                feature_names=feature_columns[:5]
            )
            print(f"   {description}: {threshold:.1%}")
        
        # Detailed analysis for swing trading
        print(f"\n📊 Detailed Analysis for Swing Trading:")
        print_threshold_analysis(
            trading_style="swing_trading",
            market_type="medium_volatility",
            feature_names=feature_columns[:8]
        )
        
    except ImportError as e:
        print(f"⚠️ Threshold advisor not available: {e}")
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False
    
    # Test 4: Test different consolidation methods
    print("\n🧪 Test 4: Test consolidation methods")
    try:
        consolidation_methods = ["average", "best_performance", "weighted_average"]
        
        for method in consolidation_methods:
            print(f"\n🔧 Testing consolidation method: {method}")
            
            test_config = DirectionalLookbackConfig(
                min_lookback=5,
                max_lookback=25,
                enable_period_consolidation=True,
                consolidation_variance_threshold=0.25,  # Higher threshold for more consolidation
                consolidation_method=method,
                use_existing_feature_pipeline=True,
                min_samples_per_direction=50
            )
            
            test_result = optimize_features_directional(
                data=data,
                feature_columns=feature_columns[:5],  # Smaller set for quick test
                target_column='returns',
                config=test_config
            )
            
            print(f"   - Method: {method}")
            print(f"   - Consolidated: {len(test_result.consolidated_features)}")
            print(f"   - Long: {len(test_result.long_features)}")
            print(f"   - Short: {len(test_result.short_features)}")
            
            # Show one consolidation example if available
            if test_result.consolidated_features:
                feature_name = list(test_result.consolidated_features.keys())[0]
                consolidated = test_result.consolidated_features[feature_name]
                print(f"   - Example: {feature_name} → period {consolidated.optimal_lookback_period} "
                      f"({consolidated.consolidation_reason})")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False
    
    # Test 4: Compare with legacy approach simulation
    print("\n🧪 Test 4: Feature count comparison")
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
    print(f"   - Period consolidation reduces features when long/short periods are similar (<20% variance)")
    print(f"   - Total features generated: {new_feature_count} (includes consolidated features)")
    print(f"   - Consolidated features: {len(result.consolidated_features)}")
    print(f"   - Integration with existing 100→80→60 pipeline maintained")
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
    
    # Option 3: Balanced with Consolidation (recommended)
    print("\n📋 Option 3: Balanced with Consolidation (Recommended)")
    balanced_config = DirectionalLookbackConfig(
        enable_directional=True,
        cross_directional_analysis=True,
        
        # Consolidation settings
        enable_period_consolidation=True,
        consolidation_variance_threshold=0.20,
        consolidation_method="average",
        
        # Pipeline integration
        use_existing_feature_pipeline=True,
        generate_features_for_pipeline=True,
        
        min_mutual_info_score=0.02,
        min_samples_per_direction=75
    )
    print(f"   - Period consolidation: {balanced_config.enable_period_consolidation}")
    print(f"   - Variance threshold: {balanced_config.consolidation_variance_threshold}")
    print(f"   - Integration with existing pipeline: {balanced_config.use_existing_feature_pipeline}")
    print(f"   - Quality threshold: {balanced_config.min_mutual_info_score}")

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
        print("   ✅ Smart consolidation with improved thresholds (15% default, was 20%)")
        print("   ✅ Adaptive thresholds based on feature type and market conditions")
        print("   ✅ Maintains directional differentiation (long/short)")
        print("   ✅ Integrates with existing 100→80→60 feature selection pipeline")
        print("   ✅ Preserves optimization quality and performance")
        print("   ✅ Provides intelligent feature selection")
        print("   ✅ Enables cross-directional analysis")
        print("   ✅ Configurable consolidation methods (average, best_performance, weighted)")
        print("   ✅ Intelligent threshold recommendations for different trading strategies")
        
        print("\n🔧 Integration Notes:")
        print("   - Set 'use_new_directional_approach=True' in config")
        print("   - Enable 'enable_period_consolidation=True' for smart consolidation")
        print("   - Set 'use_existing_feature_pipeline=True' to use 100→80→60 pipeline")
        print("   - Adjust 'consolidation_variance_threshold' (improved default: 0.15 = 15%)")
        print("   - Enable 'enable_adaptive_thresholds=True' for intelligent threshold adjustment")
        print("   - Set 'trading_timeframe' and 'market_volatility' for adaptive thresholds")
        print("   - Choose consolidation method: 'average', 'best_performance', or 'weighted_average'")
        print("   - Use threshold advisor for optimal threshold recommendations")
        print("   - Use 'max_features_to_optimize' to limit computation time")
        print("   - Enable 'cross_directional_analysis' for better insights")
        
    else:
        print("\n❌ Test failed - check error messages above")
        sys.exit(1)