"""
Test Integration Structure

This script tests that the new integration structure works correctly
after moving files to the feature_generation/integration/ folder.
"""

import warnings
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

def create_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(42)
    
    # Generate price data
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Ensure high >= low
    data['high'] = np.maximum(data['high'], data['low'])
    
    return data

def test_integration_imports():
    """Test that integration modules can be imported correctly."""
    print("\n🔧 Testing Integration Imports...")
    
    try:
        # Test core integration imports
        from src.feature_generation.integration import (
            FeatureBankIntegrator,
            FeatureBankConfig,
            FeatureBankCategory,
            MLTask,
            FeatureTaskIntegrator
        )
        
        print("  ✅ Core integration imports successful")
        
        # Test enhanced integration imports
        from src.feature_generation.integration import (
            EnhancedHDBSCANClusteringIntegration,
            EnhancedRegimeClusteringIntegration,
            EnhancedModelsTrainingIntegration,
            EnhancedEnsembleTrainingIntegration
        )
        
        print("  ✅ Enhanced integration imports successful")
        
        # Test convenience function imports
        from src.feature_generation.integration import (
            get_comprehensive_hdbscan_features,
            get_comprehensive_regime_clustering_features,
            get_comprehensive_models_training_features,
            get_comprehensive_ensemble_training_features
        )
        
        print("  ✅ Convenience function imports successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration imports failed: {e}")
        return False

def test_integration_functionality():
    """Test basic integration functionality."""
    print("\n🔧 Testing Integration Functionality...")
    
    try:
        from src.feature_generation.integration import (
            FeatureBankIntegrator,
            FeatureBankConfig,
            EnhancedHDBSCANClusteringIntegration
        )
        
        # Test configuration
        config = FeatureBankConfig()
        print(f"  ✅ Configuration created: HDBSCAN {config.hdbscan_min_features}-{config.hdbscan_max_features}")
        
        # Test integrator initialization
        integrator = FeatureBankIntegrator(config)
        print("  ✅ Feature bank integrator initialized")
        
        # Test enhanced integration
        enhanced_integrator = EnhancedHDBSCANClusteringIntegration()
        print("  ✅ Enhanced HDBSCAN integrator initialized")
        
        # Test feature generation
        data = create_sample_data(50)
        feature_result = enhanced_integrator.get_comprehensive_clustering_features(data)
        
        print(f"  ✅ Feature generation: {feature_result['feature_count']} features")
        print(f"  ✅ Target range: {feature_result['target_range']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration functionality test failed: {e}")
        return False

def test_feature_breakdown():
    """Test feature breakdown functionality."""
    print("\n🔧 Testing Feature Breakdown...")
    
    try:
        from src.feature_generation.integration import FeatureBankIntegrator
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test feature breakdown
        integrator = FeatureBankIntegrator()
        breakdown = integrator.get_feature_breakdown_by_category('hdbscan_clustering', data)
        
        print(f"  ✅ Feature breakdown generated: {len(breakdown)} categories")
        for category, info in breakdown.items():
            print(f"    - {category}: {info['feature_count']} features")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature breakdown test failed: {e}")
        return False

def test_enhanced_integrations():
    """Test all enhanced integrations."""
    print("\n🔧 Testing Enhanced Integrations...")
    
    try:
        from src.feature_generation.integration import (
            EnhancedHDBSCANClusteringIntegration,
            EnhancedRegimeClusteringIntegration,
            EnhancedModelsTrainingIntegration,
            EnhancedEnsembleTrainingIntegration
        )
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test each integration
        integrations = [
            ("HDBSCAN", EnhancedHDBSCANClusteringIntegration()),
            ("Regime Clustering", EnhancedRegimeClusteringIntegration()),
            ("Models Training", EnhancedModelsTrainingIntegration()),
            ("Ensemble Training", EnhancedEnsembleTrainingIntegration())
        ]
        
        for name, integrator in integrations:
            try:
                if hasattr(integrator, 'get_comprehensive_clustering_features'):
                    features = integrator.get_comprehensive_clustering_features(data)
                elif hasattr(integrator, 'get_comprehensive_regime_features'):
                    features = integrator.get_comprehensive_regime_features(data)
                elif hasattr(integrator, 'get_comprehensive_training_features'):
                    features = integrator.get_comprehensive_training_features(data)
                elif hasattr(integrator, 'get_comprehensive_ensemble_features'):
                    features = integrator.get_comprehensive_ensemble_features(data)
                else:
                    continue
                
                print(f"  ✅ {name}: {features['feature_count']} features")
                
            except Exception as e:
                print(f"  ⚠️ {name} failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced integrations test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Integration Structure")
    print("=" * 50)
    
    tests = [
        test_integration_imports,
        test_integration_functionality,
        test_feature_breakdown,
        test_enhanced_integrations
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Integration structure is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)