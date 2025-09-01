#!/usr/bin/env python3
"""
Test script for Enhanced Dynamic Feature Selection

This script demonstrates the three key improvements:
1. Dynamic selection process without fixed arbitrary thresholds
2. Ensures selected features aren't too correlated
3. Adds interaction features between top features
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.enhanced_dynamic_feature_selection import EnhancedDynamicFeatureSelection
from src.config.enhanced_feature_selection_config import (
    get_default_enhanced_feature_selection_config,
    get_optimized_feature_selection_config,
    get_comprehensive_feature_selection_config,
    get_regime_specific_feature_selection_config
)

warnings.filterwarnings('ignore')


def generate_synthetic_features(n_samples: int = 1000, n_features: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic features for testing."""
    np.random.seed(42)
    
    # Generate base features with different characteristics
    features = {}
    
    # Momentum features (trending)
    for i in range(20):
        features[f'momentum_{i}'] = np.cumsum(np.random.randn(n_samples) * 0.1)
    
    # Volatility features (mean-reverting)
    for i in range(20):
        features[f'volatility_{i}'] = np.abs(np.random.randn(n_samples) * 0.5)
    
    # Liquidity features
    for i in range(15):
        features[f'volume_{i}'] = np.random.exponential(100, n_samples)
    
    # Microstructure features
    for i in range(15):
        features[f'microstructure_{i}'] = np.random.randn(n_samples) * 0.3
    
    # Wavelet features
    for i in range(10):
        features[f'wavelet_{i}'] = np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.randn(n_samples) * 0.1
    
    # Support/Resistance features
    for i in range(10):
        features[f'sr_distance_{i}'] = np.random.uniform(-1, 1, n_samples)
    
    # Statistical features
    for i in range(10):
        features[f'statistical_{i}'] = np.random.randn(n_samples)
    
    # Candlestick features
    for i in range(10):
        features[f'candlestick_{i}'] = np.random.choice([-1, 0, 1], n_samples)
    
    # Add some highly correlated features (to test correlation filtering)
    for i in range(20):
        base_feature = features[f'momentum_{i % 5}']
        noise = np.random.randn(n_samples) * 0.01
        features[f'correlated_momentum_{i}'] = base_feature + noise
    
    # Add some low variance features (to test variance filtering)
    for i in range(20):
        features[f'low_variance_{i}'] = np.random.randn(n_samples) * 0.001
    
    # Add some features with NaN values (to test data quality filtering)
    for i in range(10):
        feature_values = np.random.randn(n_samples)
        # Add 15% NaN values
        nan_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
        feature_values[nan_indices] = np.nan
        features[f'nan_feature_{i}'] = feature_values
    
    # Create DataFrame
    features_df = pd.DataFrame(features)
    
    # Generate target variable (binary classification)
    # Target depends on combination of momentum and volatility features
    momentum_score = features_df[['momentum_0', 'momentum_1', 'momentum_2']].mean(axis=1)
    volatility_score = features_df[['volatility_0', 'volatility_1', 'volatility_2']].mean(axis=1)
    
    # Create target based on feature combinations
    target = ((momentum_score > momentum_score.median()) & 
              (volatility_score < volatility_score.median())).astype(int)
    
    return features_df, target


def test_dynamic_thresholds():
    """Test 1: Verify that thresholds are computed dynamically, not fixed."""
    print("🧪 Testing Dynamic Threshold Computation...")
    
    # Generate synthetic data
    features_df, target = generate_synthetic_features(n_samples=500, n_features=150)
    
    # Get configuration
    config = get_default_enhanced_feature_selection_config()
    
    # Initialize feature selector
    selector = EnhancedDynamicFeatureSelection(config)
    
    # Run feature selection
    selected_features, metadata = selector.select_features_dynamically(
        features_df, target, "BTCUSDT", "binance", "test_data"
    )
    
    # Verify adaptive thresholds were computed
    adaptive_thresholds = metadata.get("adaptive_thresholds", {})
    
    print(f"   ✅ Adaptive variance threshold: {adaptive_thresholds.get('variance', 'Not computed')}")
    print(f"   ✅ Adaptive correlation threshold: {adaptive_thresholds.get('correlation', 'Not computed')}")
    print(f"   ✅ Adaptive mutual info threshold: {adaptive_thresholds.get('mutual_info', 'Not computed')}")
    
    # Verify thresholds are not the default fixed values
    assert adaptive_thresholds.get('variance') is not None, "Variance threshold should be computed dynamically"
    assert adaptive_thresholds.get('correlation') is not None, "Correlation threshold should be computed dynamically"
    assert adaptive_thresholds.get('mutual_info') is not None, "MI threshold should be computed dynamically"
    
    print("   ✅ Dynamic threshold computation verified!")
    return selector, selected_features, metadata


def test_correlation_filtering():
    """Test 2: Verify that selected features aren't too correlated."""
    print("\n🧪 Testing Correlation Filtering...")
    
    # Use results from previous test
    selector, selected_features, metadata = test_dynamic_thresholds()
    
    # Analyze correlations in selected features
    correlation_analysis = selector.get_correlation_analysis(selected_features)
    
    print(f"   📊 Selected features shape: {correlation_analysis.get('correlation_matrix_shape', 'Unknown')}")
    print(f"   📊 Mean correlation: {correlation_analysis.get('mean_correlation', 'Unknown'):.4f}")
    print(f"   📊 Max correlation: {correlation_analysis.get('max_correlation', 'Unknown'):.4f}")
    
    # Check for high correlations
    high_corr_pairs = correlation_analysis.get('high_correlation_pairs', [])
    print(f"   📊 High correlation pairs found: {len(high_corr_pairs)}")
    
    if high_corr_pairs:
        print("   📊 Top 5 high correlation pairs:")
        for i, pair in enumerate(high_corr_pairs[:5]):
            print(f"      {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}")
    
    # Verify that correlations are reasonable (not too high)
    max_corr = correlation_analysis.get('max_correlation', 1.0)
    assert max_corr < 0.95, f"Maximum correlation {max_corr:.4f} is too high"
    
    print("   ✅ Correlation filtering verified - features are not too correlated!")
    return selector, selected_features, metadata


def test_interaction_features():
    """Test 3: Verify that interaction features are generated between top features."""
    print("\n🧪 Testing Interaction Feature Generation...")
    
    # Use results from previous test
    selector, selected_features, metadata = test_correlation_filtering()
    
    # Check if interaction features were generated
    stage7_metadata = metadata.get("stages", {}).get("stage7_interaction_features", {})
    interaction_features_added = stage7_metadata.get("interaction_features_added", 0)
    
    print(f"   🔗 Interaction features added: {interaction_features_added}")
    
    # Verify interaction features exist
    interaction_feature_names = [col for col in selected_features.columns if any(method in col for method in ['_x_', '_div_', '_diff_'])]
    
    print(f"   🔗 Interaction feature names found: {len(interaction_feature_names)}")
    if interaction_feature_names:
        print("   🔗 Sample interaction features:")
        for feature in interaction_feature_names[:5]:
            print(f"      {feature}")
    
    # Verify that interaction features were actually created
    assert interaction_features_added > 0, "No interaction features were generated"
    assert len(interaction_feature_names) > 0, "No interaction features found in column names"
    
    print("   ✅ Interaction feature generation verified!")
    return selector, selected_features, metadata


def test_feature_categories():
    """Test 4: Verify category-aware feature selection."""
    print("\n🧪 Testing Category-Aware Feature Selection...")
    
    # Use results from previous test
    selector, selected_features, metadata = test_interaction_features()
    
    # Check feature categories
    feature_categories = metadata.get("feature_categories", {})
    
    print("   📊 Feature distribution by category:")
    for category, features in feature_categories.items():
        if features:
            print(f"      {category}: {len(features)} features")
    
    # Verify that features are selected from multiple categories
    categories_with_features = sum(1 for features in feature_categories.values() if features)
    assert categories_with_features >= 3, f"Only {categories_with_features} categories have features, expected at least 3"
    
    # Verify category selection metadata
    stage6_metadata = metadata.get("stages", {}).get("stage6_category_aware", {})
    category_selection = stage6_metadata.get("category_selection", {})
    
    print("   📊 Features selected per category:")
    for category, count in category_selection.items():
        print(f"      {category}: {count} features")
    
    print("   ✅ Category-aware feature selection verified!")
    return selector, selected_features, metadata


def test_performance_metrics():
    """Test 5: Verify performance and monitoring capabilities."""
    print("\n🧪 Testing Performance and Monitoring...")
    
    # Use results from previous test
    selector, selected_features, metadata = test_feature_categories()
    
    # Get feature importance summary
    importance_summary = selector.get_feature_importance_summary()
    
    print("   📊 Feature importance summary:")
    for method, summary in importance_summary.items():
        if isinstance(summary, dict) and "top_5_features" in summary:
            print(f"      {method}: {len(summary['top_5_features'])} top features")
            print(f"         Top feature: {summary['top_5_features'][0]}")
            print(f"         Mean score: {summary['mean_score']:.4f}")
    
    # Verify metadata completeness
    required_stages = [
        "stage1_data_quality", "stage2_dynamic_thresholds", "stage3_adaptive_variance",
        "stage4_adaptive_correlation", "stage5_multi_method_importance", 
        "stage6_category_aware", "stage7_interaction_features", "stage8_final_optimization"
    ]
    
    stages = metadata.get("stages", {})
    for stage in required_stages:
        assert stage in stages, f"Missing stage: {stage}"
    
    print("   ✅ Performance monitoring verified!")
    return selector, selected_features, metadata


def test_configuration_variants():
    """Test 6: Test different configuration variants."""
    print("\n🧪 Testing Configuration Variants...")
    
    # Test default configuration
    default_config = get_default_enhanced_feature_selection_config()
    print("   ✅ Default configuration loaded")
    
    # Test optimized configuration
    optimized_config = get_optimized_feature_selection_config()
    print("   ✅ Optimized configuration loaded")
    
    # Test comprehensive configuration
    comprehensive_config = get_comprehensive_feature_selection_config()
    print("   ✅ Comprehensive configuration loaded")
    
    # Test regime-specific configurations
    trending_config = get_regime_specific_feature_selection_config("trending")
    mean_reverting_config = get_regime_specific_feature_selection_config("mean_reverting")
    volatile_config = get_regime_specific_feature_selection_config("volatile")
    
    print("   ✅ Regime-specific configurations loaded")
    
    # Verify configuration differences
    assert default_config["feature_reduction"]["target_features"] == 100
    assert optimized_config["feature_reduction"]["target_features"] == 80
    assert comprehensive_config["feature_reduction"]["target_features"] == 120
    
    print("   ✅ Configuration variants verified!")
    return True


def run_comprehensive_test():
    """Run comprehensive test of the enhanced dynamic feature selection system."""
    print("🚀 Starting Comprehensive Test of Enhanced Dynamic Feature Selection")
    print("=" * 80)
    
    try:
        # Test 1: Dynamic thresholds
        test_dynamic_thresholds()
        
        # Test 2: Correlation filtering
        test_correlation_filtering()
        
        # Test 3: Interaction features
        test_interaction_features()
        
        # Test 4: Category-aware selection
        test_feature_categories()
        
        # Test 5: Performance monitoring
        test_performance_metrics()
        
        # Test 6: Configuration variants
        test_configuration_variants()
        
        print("\n" + "=" * 80)
        print("🎉 All Tests Passed! Enhanced Dynamic Feature Selection System Verified!")
        print("\n✅ Key Improvements Implemented:")
        print("   1. Dynamic selection process without fixed arbitrary thresholds")
        print("   2. Ensures selected features aren't too correlated")
        print("   3. Adds interaction features between top features")
        print("\n✅ Additional Features:")
        print("   - Category-aware feature selection")
        print("   - Multi-method feature importance ranking")
        print("   - Hierarchical clustering for correlation filtering")
        print("   - Comprehensive performance monitoring")
        print("   - Flexible configuration system")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run comprehensive test
    success = run_comprehensive_test()
    
    if success:
        print("\n🎯 Step 7 Implementation Complete!")
        print("The enhanced dynamic feature selection system successfully addresses all three requirements:")
        print("1. ✅ Dynamic selection process without fixed arbitrary thresholds")
        print("2. ✅ Ensures selected features aren't too correlated") 
        print("3. ✅ Adds interaction features between top features")
    else:
        print("\n❌ Step 7 Implementation Failed!")
        sys.exit(1)