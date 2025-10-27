"""
Minimal Test for Enhanced Integration System

This script tests only the core functionality of the enhanced integration system
without any complex dependencies that might cause import issues.
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

def test_feature_bank_config():
    """Test feature bank configuration."""
    print("\n🔧 Testing Feature Bank Configuration...")
    
    try:
        from src.feature_generation.categories.feature_bank_integration import FeatureBankConfig, FeatureBankCategory
        
        # Test configuration creation
        config = FeatureBankConfig()
        
        print(f"  ✅ HDBSCAN range: {config.hdbscan_min_features}-{config.hdbscan_max_features}")
        print(f"  ✅ Regime clustering range: {config.regime_clustering_min_features}-{config.regime_clustering_max_features}")
        print(f"  ✅ Models training range: {config.models_training_min_features}-{config.models_training_max_features}")
        print(f"  ✅ Ensemble training range: {config.ensemble_training_min_features}-{config.ensemble_training_max_features}")
        
        # Test weights
        print(f"  ✅ HDBSCAN weights: {config.hdbscan_weights}")
        print(f"  ✅ Regime clustering weights: {config.regime_clustering_weights}")
        print(f"  ✅ Models training weights: {config.models_training_weights}")
        print(f"  ✅ Ensemble training weights: {config.ensemble_training_weights}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature bank configuration test failed: {e}")
        return False

def test_enhanced_integration_classes():
    """Test enhanced integration classes."""
    print("\n🔧 Testing Enhanced Integration Classes...")
    
    try:
        # Test HDBSCAN integration
        from src.feature_generation.categories.enhanced_hdbscan_clustering_integration import EnhancedHDBSCANClusteringIntegration
        
        hdbscan_integrator = EnhancedHDBSCANClusteringIntegration()
        print("  ✅ Enhanced HDBSCAN integration class created")
        
        # Test regime clustering integration
        from src.feature_generation.categories.enhanced_regime_clustering_integration import EnhancedRegimeClusteringIntegration
        
        regime_integrator = EnhancedRegimeClusteringIntegration()
        print("  ✅ Enhanced regime clustering integration class created")
        
        # Test models training integration
        from src.feature_generation.categories.enhanced_models_training_integration import EnhancedModelsTrainingIntegration
        
        models_integrator = EnhancedModelsTrainingIntegration()
        print("  ✅ Enhanced models training integration class created")
        
        # Test ensemble training integration
        from src.feature_generation.categories.enhanced_ensemble_training_integration import EnhancedEnsembleTrainingIntegration
        
        ensemble_integrator = EnhancedEnsembleTrainingIntegration()
        print("  ✅ Enhanced ensemble training integration class created")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced integration classes test failed: {e}")
        return False

def test_feature_generation_basic():
    """Test basic feature generation."""
    print("\n🔧 Testing Basic Feature Generation...")
    
    try:
        from src.feature_generation.categories.enhanced_hdbscan_clustering_integration import EnhancedHDBSCANClusteringIntegration
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test feature generation
        integrator = EnhancedHDBSCANClusteringIntegration()
        feature_result = integrator.get_comprehensive_clustering_features(data)
        
        print(f"  ✅ Feature generation: {feature_result['feature_count']} features")
        print(f"  ✅ Target range: {feature_result['target_range']}")
        print(f"  ✅ Clustering optimized: {feature_result.get('clustering_optimized', False)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic feature generation test failed: {e}")
        return False

def test_data_preparation_basic():
    """Test basic data preparation."""
    print("\n🔧 Testing Basic Data Preparation...")
    
    try:
        from src.feature_generation.categories.enhanced_hdbscan_clustering_integration import EnhancedHDBSCANClusteringIntegration
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test data preparation
        integrator = EnhancedHDBSCANClusteringIntegration()
        feature_matrix, feature_names, metadata = integrator.prepare_data_for_clustering(data)
        
        print(f"  ✅ Feature matrix shape: {feature_matrix.shape}")
        print(f"  ✅ Feature names count: {len(feature_names)}")
        print(f"  ✅ Metadata keys: {list(metadata.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic data preparation test failed: {e}")
        return False

def test_feature_category_breakdown_basic():
    """Test feature category breakdown."""
    print("\n🔧 Testing Feature Category Breakdown...")
    
    try:
        from src.feature_generation.categories.enhanced_hdbscan_clustering_integration import EnhancedHDBSCANClusteringIntegration
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test feature generation
        integrator = EnhancedHDBSCANClusteringIntegration()
        feature_result = integrator.get_comprehensive_clustering_features(data)
        
        if feature_result['features']:
            breakdown = integrator._get_feature_category_breakdown(feature_result['features'])
            print(f"  ✅ Feature breakdown: {breakdown}")
        else:
            print("  ⚠️ No features generated for breakdown analysis")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature category breakdown test failed: {e}")
        return False

def test_clustering_readiness_assessment():
    """Test clustering readiness assessment."""
    print("\n🔧 Testing Clustering Readiness Assessment...")
    
    try:
        from src.feature_generation.categories.enhanced_hdbscan_clustering_integration import EnhancedHDBSCANClusteringIntegration
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test feature generation
        integrator = EnhancedHDBSCANClusteringIntegration()
        feature_result = integrator.get_comprehensive_clustering_features(data)
        
        if feature_result['features']:
            readiness = integrator._assess_clustering_readiness(feature_result['features'])
            print(f"  ✅ Clustering readiness score: {readiness['score']}")
            print(f"  ✅ Issues: {len(readiness['issues'])}")
            print(f"  ✅ Feature count: {readiness['feature_count']}")
        else:
            print("  ⚠️ No features generated for readiness assessment")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Clustering readiness assessment test failed: {e}")
        return False

def test_lgbm_shap_basic():
    """Test basic LGBM-SHAP functionality."""
    print("\n🔧 Testing Basic LGBM-SHAP...")
    
    try:
        import lightgbm as lgb
        import shap
        
        # Create sample data
        n_samples = 100
        n_features = 50
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        # Train LGBM model
        model = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)
        
        # Get SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Calculate feature importance
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Select top 10 features
        top_features = np.argsort(feature_importance)[-10:]
        
        print(f"  ✅ LGBM model trained successfully")
        print(f"  ✅ SHAP values calculated: {shap_values.shape}")
        print(f"  ✅ Feature importance calculated: {len(feature_importance)} features")
        print(f"  ✅ Top 10 features selected: {top_features}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic LGBM-SHAP test failed: {e}")
        return False

def test_ensemble_features_basic():
    """Test basic ensemble feature generation."""
    print("\n🔧 Testing Basic Ensemble Features...")
    
    try:
        from src.feature_generation.categories.enhanced_ensemble_training_integration import EnhancedEnsembleTrainingIntegration
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test ensemble integration
        integrator = EnhancedEnsembleTrainingIntegration()
        feature_result = integrator.get_comprehensive_ensemble_features(data)
        
        print(f"  ✅ Generated {feature_result['feature_count']} ensemble features")
        print(f"  ✅ Includes base outputs: {feature_result['includes_base_outputs']}")
        print(f"  ✅ Includes disagreement: {feature_result['includes_disagreement']}")
        print(f"  ✅ Includes entropy: {feature_result['includes_entropy']}")
        
        # Test synthetic target creation
        target = integrator._create_synthetic_target(data)
        print(f"  ✅ Synthetic target created: {target.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic ensemble features test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Integration System (Minimal)")
    print("=" * 60)
    
    tests = [
        test_feature_bank_config,
        test_enhanced_integration_classes,
        test_feature_generation_basic,
        test_data_preparation_basic,
        test_feature_category_breakdown_basic,
        test_clustering_readiness_assessment,
        test_lgbm_shap_basic,
        test_ensemble_features_basic
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
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Enhanced integration system is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)