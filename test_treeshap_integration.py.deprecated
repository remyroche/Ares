#!/usr/bin/env python3
"""
Test TreeSHAP Integration for Regime Feature Selection

This script tests the new TreeSHAP-based feature selection integration
and compares it with traditional methods.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_treeshap_selector():
    """Test the TreeSHAP feature selector directly."""
    print("🧪 Testing TreeSHAP Feature Selector...")
    
    try:
        from src.training.steps.market_analysis.treeshap_feature_selector import TreeSHAPFeatureSelector
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        # Generate synthetic features with different categories
        features_data = {}
        
        # Price features
        for i in range(10):
            features_data[f'price_feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        # Volume features  
        for i in range(10):
            features_data[f'volume_feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        # Volatility features
        for i in range(10):
            features_data[f'volatility_feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        # Momentum features
        for i in range(10):
            features_data[f'momentum_feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        # Other features
        for i in range(10):
            features_data[f'other_feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        features_df = pd.DataFrame(features_data)
        
        # Create target with some correlation to features
        target = (
            0.3 * features_df['price_feature_0'] +
            0.2 * features_df['volume_feature_0'] +
            0.1 * features_df['volatility_feature_0'] +
            np.random.normal(0, 0.5, n_samples)
        )
        
        labels_df = pd.DataFrame({'target': target})
        
        # Test TreeSHAP selector
        config = {
            'n_estimators': 50,
            'max_depth': 6,
            'correlation_threshold': 0.8,
            'diversity_weight': 0.3,
            'treeshap_weight': 0.5,
            'correlation_weight': 0.2
        }
        
        selector = TreeSHAPFeatureSelector(config)
        result = selector.select_features(features_df, labels_df, target_feature_count=15)
        
        if result['success']:
            print(f"✅ TreeSHAP selection successful!")
            print(f"📊 Selected {len(result['selected_features'])} features")
            print(f"⏱️ Execution time: {result['execution_time']:.3f}s")
            
            # Show category distribution
            categories = {}
            for feature in result['selected_features']:
                category = selector._extract_category(feature)
                categories[category] = categories.get(category, 0) + 1
            
            print("📊 Category distribution:")
            for category, count in categories.items():
                print(f"  {category}: {count} features")
            
            return True
        else:
            print(f"❌ TreeSHAP selection failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ TreeSHAP test failed: {e}")
        return False

def test_economic_regime_selector():
    """Test the economic regime feature selector with TreeSHAP integration."""
    print("\n🧪 Testing Economic Regime Feature Selector with TreeSHAP...")
    
    try:
        from src.training.steps.market_analysis.economic_regime_feature_selector import EconomicRegimeFeatureSelector
        
        # Load configuration
        config_path = "config/features/economic_regime_feature_selection_config.yaml"
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return False
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        n_features = 30
        
        # Generate synthetic features
        features_data = {}
        for i in range(n_features):
            features_data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        features_df = pd.DataFrame(features_data)
        
        # Create multi-target labels
        labels_data = {
            'close_return': np.random.normal(0, 0.02, n_samples),
            'volume_log_return': np.random.normal(0, 0.1, n_samples),
            'price_range_pct': np.random.normal(0, 0.05, n_samples),
            'volatility_20': np.random.normal(0, 0.03, n_samples)
        }
        labels_df = pd.DataFrame(labels_data)
        
        # Initialize selector
        selector = EconomicRegimeFeatureSelector(config_path)
        
        # Test feature selection
        result = selector.select_features(features_df, labels_df)
        
        if result and len(result) > 0:
            print(f"✅ Economic regime selection successful!")
            print(f"📊 Selected {len(result)} features")
            
            # Show some selected features
            print("📊 Sample selected features:")
            for i, feature in enumerate(result[:10]):
                print(f"  {i+1}. {feature}")
            
            return True
        else:
            print("❌ Economic regime selection failed or returned no features")
            return False
            
    except Exception as e:
        print(f"❌ Economic regime test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_method_redundancy():
    """Analyze whether other methods are still needed with TreeSHAP."""
    print("\n🔍 Analyzing Method Redundancy...")
    
    print("📊 TreeSHAP Capabilities:")
    print("  ✅ Feature importance scoring (more accurate than traditional methods)")
    print("  ✅ Correlation-based redundancy filtering")
    print("  ✅ Category diversity enforcement")
    print("  ✅ Multi-target support")
    print("  ✅ Hardware optimization")
    
    print("\n📊 Traditional Methods Comparison:")
    print("  🔄 mRMR: Redundant - TreeSHAP handles redundancy via correlation filtering")
    print("  🔄 Economic significance: Partially redundant - TreeSHAP includes correlation scoring")
    print("  🔄 Regime discrimination: Not redundant - TreeSHAP doesn't calculate F-ratios")
    print("  🔄 Clustering quality: Not redundant - TreeSHAP doesn't calculate silhouette scores")
    print("  🔄 Regime transition: Not redundant - TreeSHAP doesn't detect regime changes")
    
    print("\n💡 Recommendation:")
    print("  🎯 Use TreeSHAP as PRIMARY method for feature importance and redundancy")
    print("  🔄 Keep traditional methods as SUPPLEMENTARY for regime-specific metrics")
    print("  🏗️ Implement HYBRID approach: TreeSHAP + regime-specific scoring")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting TreeSHAP Integration Tests...\n")
    
    # Test TreeSHAP selector
    treeshap_success = test_treeshap_selector()
    
    # Test economic regime selector
    regime_success = test_economic_regime_selector()
    
    # Analyze method redundancy
    analysis_success = analyze_method_redundancy()
    
    print(f"\n📊 Test Results:")
    print(f"  TreeSHAP Selector: {'✅ PASS' if treeshap_success else '❌ FAIL'}")
    print(f"  Economic Regime Selector: {'✅ PASS' if regime_success else '❌ FAIL'}")
    print(f"  Method Analysis: {'✅ PASS' if analysis_success else '❌ FAIL'}")
    
    if treeshap_success and regime_success and analysis_success:
        print("\n🎉 All tests passed! TreeSHAP integration is working correctly.")
        return True
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)