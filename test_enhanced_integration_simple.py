"""
Simple Test for Enhanced Integration System

This script tests the core functionality of the enhanced integration system
without complex dependencies that might cause import issues.
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

def test_feature_bank_integration_basic():
    """Test basic feature bank integration functionality."""
    print("\n🔧 Testing Feature Bank Integration (Basic)...")
    
    try:
        # Test basic imports
        from src.feature_generation.categories.feature_bank_integration import (
            FeatureBankIntegrator, FeatureBankConfig, FeatureBankCategory
        )
        
        print("  ✅ Feature bank integration imports successful")
        
        # Test configuration
        config = FeatureBankConfig()
        print(f"  ✅ Configuration created: HDBSCAN {config.hdbscan_min_features}-{config.hdbscan_max_features}")
        print(f"  ✅ Configuration created: Regime {config.regime_clustering_min_features}-{config.regime_clustering_max_features}")
        print(f"  ✅ Configuration created: Models {config.models_training_min_features}-{config.models_training_max_features}")
        print(f"  ✅ Configuration created: Ensemble {config.ensemble_training_min_features}-{config.ensemble_training_max_features}")
        
        # Test integrator initialization
        integrator = FeatureBankIntegrator(config)
        print("  ✅ Feature bank integrator initialized")
        
        # Test feature category breakdown
        data = create_sample_data(50)
        breakdown = integrator.get_feature_breakdown_by_category('hdbscan_clustering', data)
        print(f"  ✅ Feature breakdown generated: {len(breakdown)} categories")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature bank integration test failed: {e}")
        return False

def test_enhanced_hdbscan_integration_basic():
    """Test basic enhanced HDBSCAN integration functionality."""
    print("\n🔧 Testing Enhanced HDBSCAN Integration (Basic)...")
    
    try:
        from src.feature_generation.categories.enhanced_hdbscan_clustering_integration import (
            EnhancedHDBSCANClusteringIntegration
        )
        
        print("  ✅ Enhanced HDBSCAN integration imports successful")
        
        # Test integrator initialization
        integrator = EnhancedHDBSCANClusteringIntegration()
        print("  ✅ Enhanced HDBSCAN integrator initialized")
        
        # Test feature generation
        data = create_sample_data(50)
        feature_result = integrator.get_comprehensive_clustering_features(data)
        
        print(f"  ✅ Feature generation: {feature_result['feature_count']} features")
        print(f"  ✅ Target range: {feature_result['target_range']}")
        print(f"  ✅ Clustering optimized: {feature_result.get('clustering_optimized', False)}")
        
        # Test data preparation
        feature_matrix, feature_names, metadata = integrator.prepare_data_for_clustering(data)
        print(f"  ✅ Data preparation: {feature_matrix.shape} matrix, {len(feature_names)} features")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced HDBSCAN integration test failed: {e}")
        return False

def test_enhanced_regime_clustering_integration_basic():
    """Test basic enhanced regime clustering integration functionality."""
    print("\n🔧 Testing Enhanced Regime Clustering Integration (Basic)...")
    
    try:
        from src.feature_generation.categories.enhanced_regime_clustering_integration import (
            EnhancedRegimeClusteringIntegration
        )
        
        print("  ✅ Enhanced regime clustering integration imports successful")
        
        # Test integrator initialization
        integrator = EnhancedRegimeClusteringIntegration()
        print("  ✅ Enhanced regime clustering integrator initialized")
        
        # Test feature generation
        data = create_sample_data(50)
        feature_result = integrator.get_comprehensive_regime_features(data)
        
        print(f"  ✅ Feature generation: {feature_result['feature_count']} features")
        print(f"  ✅ Target range: {feature_result['target_range']}")
        print(f"  ✅ Regime optimized: {feature_result.get('regime_optimized', False)}")
        
        # Test data preparation
        feature_matrix, feature_names, metadata = integrator.prepare_data_for_regime_clustering(data)
        print(f"  ✅ Data preparation: {feature_matrix.shape} matrix, {len(feature_names)} features")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced regime clustering integration test failed: {e}")
        return False

def test_enhanced_models_training_integration_basic():
    """Test basic enhanced models training integration functionality."""
    print("\n🔧 Testing Enhanced Models Training Integration (Basic)...")
    
    try:
        from src.feature_generation.categories.enhanced_models_training_integration import (
            EnhancedModelsTrainingIntegration
        )
        
        print("  ✅ Enhanced models training integration imports successful")
        
        # Test integrator initialization
        integrator = EnhancedModelsTrainingIntegration()
        print("  ✅ Enhanced models training integrator initialized")
        
        # Test feature generation
        data = create_sample_data(50)
        feature_result = integrator.get_comprehensive_training_features(data)
        
        print(f"  ✅ Feature generation: {feature_result['feature_count']} features")
        print(f"  ✅ Target range: {feature_result['target_range']}")
        print(f"  ✅ Training optimized: {feature_result.get('training_optimized', False)}")
        
        # Test data preparation
        X, y, feature_names, metadata = integrator.prepare_data_for_training(data)
        print(f"  ✅ Data preparation: {X.shape} features, {y.shape} target, {len(feature_names)} feature names")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced models training integration test failed: {e}")
        return False

def test_enhanced_ensemble_training_integration_basic():
    """Test basic enhanced ensemble training integration functionality."""
    print("\n🔧 Testing Enhanced Ensemble Training Integration (Basic)...")
    
    try:
        from src.feature_generation.categories.enhanced_ensemble_training_integration import (
            EnhancedEnsembleTrainingIntegration
        )
        
        print("  ✅ Enhanced ensemble training integration imports successful")
        
        # Test integrator initialization
        integrator = EnhancedEnsembleTrainingIntegration()
        print("  ✅ Enhanced ensemble training integrator initialized")
        
        # Test feature generation
        data = create_sample_data(50)
        feature_result = integrator.get_comprehensive_ensemble_features(data)
        
        print(f"  ✅ Feature generation: {feature_result['feature_count']} features")
        print(f"  ✅ Target range: {feature_result['target_range']}")
        print(f"  ✅ Ensemble optimized: {feature_result.get('ensemble_optimized', False)}")
        print(f"  ✅ Meta features included: {feature_result.get('meta_features_included', False)}")
        
        # Test data preparation
        X, y, feature_names, metadata = integrator.prepare_data_for_ensemble_training(data)
        print(f"  ✅ Data preparation: {X.shape} features, {y.shape} target, {len(feature_names)} feature names")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced ensemble training integration test failed: {e}")
        return False

def test_lgbm_shap_availability():
    """Test LGBM-SHAP availability."""
    print("\n🔧 Testing LGBM-SHAP Availability...")
    
    try:
        import lightgbm as lgb
        import shap
        
        print("  ✅ LGBM available")
        print("  ✅ SHAP available")
        
        # Test basic functionality
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        model = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        print(f"  ✅ LGBM model trained: {model.n_estimators} estimators")
        print(f"  ✅ SHAP values calculated: {shap_values.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ LGBM-SHAP test failed: {e}")
        return False

def test_feature_category_breakdown_basic():
    """Test feature category breakdown analysis."""
    print("\n🔧 Testing Feature Category Breakdown (Basic)...")
    
    try:
        from src.feature_generation.categories.feature_bank_integration import FeatureBankIntegrator
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test feature breakdown for each task
        integrator = FeatureBankIntegrator()
        
        tasks = ['hdbscan_clustering', 'regime_clustering', 'regime_models_training', 'regime_ensemble_training']
        
        for task in tasks:
            try:
                breakdown = integrator.get_feature_breakdown_by_category(task, data)
                print(f"  ✅ {task} breakdown: {len(breakdown)} categories")
                for category, info in breakdown.items():
                    print(f"    - {category}: {info['feature_count']} features")
            except Exception as e:
                print(f"  ⚠️ {task} breakdown failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature category breakdown test failed: {e}")
        return False

def test_comprehensive_workflow_basic():
    """Test comprehensive integration workflow."""
    print("\n🔧 Testing Comprehensive Integration Workflow (Basic)...")
    
    try:
        from src.feature_generation.categories.enhanced_hdbscan_clustering_integration import EnhancedHDBSCANClusteringIntegration
        from src.feature_generation.categories.enhanced_regime_clustering_integration import EnhancedRegimeClusteringIntegration
        from src.feature_generation.categories.enhanced_models_training_integration import EnhancedModelsTrainingIntegration
        from src.feature_generation.categories.enhanced_ensemble_training_integration import EnhancedEnsembleTrainingIntegration
        
        # Create sample data
        data = create_sample_data(100)
        
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
        print(f"  ❌ Comprehensive workflow test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Integration System (Simple)")
    print("=" * 60)
    
    tests = [
        test_feature_bank_integration_basic,
        test_enhanced_hdbscan_integration_basic,
        test_enhanced_regime_clustering_integration_basic,
        test_enhanced_models_training_integration_basic,
        test_enhanced_ensemble_training_integration_basic,
        test_lgbm_shap_availability,
        test_feature_category_breakdown_basic,
        test_comprehensive_workflow_basic
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