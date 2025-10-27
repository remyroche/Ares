"""
Test Enhanced Integration System

This script tests the comprehensive feature bank integration system that combines
existing feature bank features with regime-specific features for each ML task.

Tests:
1. Feature Bank Integration
2. Enhanced HDBSCAN Clustering Integration
3. Enhanced Regime Clustering Integration
4. Enhanced Models Training Integration
5. Enhanced Ensemble Training Integration
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

def test_feature_bank_integration():
    """Test feature bank integration."""
    print("\n🔧 Testing Feature Bank Integration...")
    
    try:
        from src.feature_generation.categories.feature_bank_integration import (
            FeatureBankIntegrator, FeatureBankConfig, FeatureBankCategory,
            get_comprehensive_hdbscan_features, get_comprehensive_regime_clustering_features,
            get_comprehensive_models_training_features, get_comprehensive_ensemble_training_features
        )
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test HDBSCAN features
        hdbscan_result = get_comprehensive_hdbscan_features(data)
        print(f"  ✅ HDBSCAN features: {hdbscan_result['feature_count']} features")
        print(f"  ✅ Target range: {hdbscan_result['target_range']}")
        
        # Test regime clustering features
        regime_result = get_comprehensive_regime_clustering_features(data)
        print(f"  ✅ Regime clustering features: {regime_result['feature_count']} features")
        print(f"  ✅ Target range: {regime_result['target_range']}")
        
        # Test models training features
        models_result = get_comprehensive_models_training_features(data)
        print(f"  ✅ Models training features: {models_result['feature_count']} features")
        print(f"  ✅ Target range: {models_result['target_range']}")
        
        # Test ensemble training features
        ensemble_result = get_comprehensive_ensemble_training_features(data)
        print(f"  ✅ Ensemble training features: {ensemble_result['feature_count']} features")
        print(f"  ✅ Target range: {ensemble_result['target_range']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature bank integration test failed: {e}")
        return False

def test_enhanced_hdbscan_integration():
    """Test enhanced HDBSCAN clustering integration."""
    print("\n🔧 Testing Enhanced HDBSCAN Clustering Integration...")
    
    try:
        from src.feature_generation.categories.enhanced_hdbscan_clustering_integration import (
            EnhancedHDBSCANClusteringIntegration,
            get_enhanced_hdbscan_features,
            perform_enhanced_hdbscan_clustering
        )
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test feature generation
        integrator = EnhancedHDBSCANClusteringIntegration()
        feature_result = integrator.get_comprehensive_clustering_features(data)
        
        print(f"  ✅ Generated {feature_result['feature_count']} clustering features")
        print(f"  ✅ Target range: {feature_result['target_range']}")
        print(f"  ✅ Clustering optimized: {feature_result.get('clustering_optimized', False)}")
        print(f"  ✅ Comprehensive features: {feature_result.get('comprehensive_features', False)}")
        
        # Test data preparation
        feature_matrix, feature_names, metadata = integrator.prepare_data_for_clustering(data)
        print(f"  ✅ Feature matrix shape: {feature_matrix.shape}")
        print(f"  ✅ Feature names count: {len(feature_names)}")
        
        # Test clustering readiness assessment
        if feature_result['features']:
            readiness = integrator._assess_clustering_readiness(feature_result['features'])
            print(f"  ✅ Clustering readiness score: {readiness['score']}")
            print(f"  ✅ Issues: {len(readiness['issues'])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced HDBSCAN integration test failed: {e}")
        return False

def test_enhanced_regime_clustering_integration():
    """Test enhanced regime clustering integration."""
    print("\n🔧 Testing Enhanced Regime Clustering Integration...")
    
    try:
        from src.feature_generation.categories.enhanced_regime_clustering_integration import (
            EnhancedRegimeClusteringIntegration,
            get_enhanced_regime_clustering_features,
            perform_enhanced_regime_clustering
        )
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test feature generation
        integrator = EnhancedRegimeClusteringIntegration()
        feature_result = integrator.get_comprehensive_regime_features(data)
        
        print(f"  ✅ Generated {feature_result['feature_count']} regime features")
        print(f"  ✅ Target range: {feature_result['target_range']}")
        print(f"  ✅ Regime optimized: {feature_result.get('regime_optimized', False)}")
        print(f"  ✅ Comprehensive features: {feature_result.get('comprehensive_features', False)}")
        
        # Test data preparation
        feature_matrix, feature_names, metadata = integrator.prepare_data_for_regime_clustering(data)
        print(f"  ✅ Feature matrix shape: {feature_matrix.shape}")
        print(f"  ✅ Feature names count: {len(feature_names)}")
        
        # Test regime readiness assessment
        if feature_result['features']:
            readiness = integrator._assess_regime_readiness(feature_result['features'])
            print(f"  ✅ Regime readiness score: {readiness['score']}")
            print(f"  ✅ Issues: {len(readiness['issues'])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced regime clustering integration test failed: {e}")
        return False

def test_enhanced_models_training_integration():
    """Test enhanced models training integration."""
    print("\n🔧 Testing Enhanced Models Training Integration...")
    
    try:
        from src.feature_generation.categories.enhanced_models_training_integration import (
            EnhancedModelsTrainingIntegration,
            get_enhanced_training_features,
            train_enhanced_models
        )
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test feature generation
        integrator = EnhancedModelsTrainingIntegration()
        feature_result = integrator.get_comprehensive_training_features(data)
        
        print(f"  ✅ Generated {feature_result['feature_count']} training features")
        print(f"  ✅ Target range: {feature_result['target_range']}")
        print(f"  ✅ Training optimized: {feature_result.get('training_optimized', False)}")
        print(f"  ✅ Comprehensive features: {feature_result.get('comprehensive_features', False)}")
        
        # Test data preparation
        X, y, feature_names, metadata = integrator.prepare_data_for_training(data)
        print(f"  ✅ Feature matrix shape: {X.shape}")
        print(f"  ✅ Target shape: {y.shape}")
        print(f"  ✅ Feature names count: {len(feature_names)}")
        
        # Test training readiness assessment
        if feature_result['features']:
            readiness = integrator._assess_training_readiness(feature_result['features'])
            print(f"  ✅ Training readiness score: {readiness['score']}")
            print(f"  ✅ Issues: {len(readiness['issues'])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced models training integration test failed: {e}")
        return False

def test_enhanced_ensemble_training_integration():
    """Test enhanced ensemble training integration."""
    print("\n🔧 Testing Enhanced Ensemble Training Integration...")
    
    try:
        from src.feature_generation.categories.enhanced_ensemble_training_integration import (
            EnhancedEnsembleTrainingIntegration,
            get_enhanced_ensemble_features,
            train_enhanced_ensemble
        )
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test feature generation
        integrator = EnhancedEnsembleTrainingIntegration()
        feature_result = integrator.get_comprehensive_ensemble_features(data)
        
        print(f"  ✅ Generated {feature_result['feature_count']} ensemble features")
        print(f"  ✅ Target range: {feature_result['target_range']}")
        print(f"  ✅ Ensemble optimized: {feature_result.get('ensemble_optimized', False)}")
        print(f"  ✅ Comprehensive features: {feature_result.get('comprehensive_features', False)}")
        print(f"  ✅ Meta features included: {feature_result.get('meta_features_included', False)}")
        
        # Test data preparation
        X, y, feature_names, metadata = integrator.prepare_data_for_ensemble_training(data)
        print(f"  ✅ Feature matrix shape: {X.shape}")
        print(f"  ✅ Target shape: {y.shape}")
        print(f"  ✅ Feature names count: {len(feature_names)}")
        
        # Test ensemble readiness assessment
        if feature_result['features']:
            readiness = integrator._assess_ensemble_readiness(feature_result['features'])
            print(f"  ✅ Ensemble readiness score: {readiness['score']}")
            print(f"  ✅ Issues: {len(readiness['issues'])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced ensemble training integration test failed: {e}")
        return False

def test_feature_category_breakdown():
    """Test feature category breakdown analysis."""
    print("\n🔧 Testing Feature Category Breakdown...")
    
    try:
        from src.feature_generation.categories.feature_bank_integration import FeatureBankIntegrator
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test feature breakdown for each task
        integrator = FeatureBankIntegrator()
        
        tasks = ['hdbscan_clustering', 'regime_clustering', 'regime_models_training', 'regime_ensemble_training']
        
        for task in tasks:
            breakdown = integrator.get_feature_breakdown_by_category(task, data)
            print(f"  ✅ {task} breakdown:")
            for category, info in breakdown.items():
                print(f"    - {category}: {info['feature_count']} features, {info['successful_generators']} generators")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature category breakdown test failed: {e}")
        return False

def test_lgbm_shap_integration():
    """Test LGBM-SHAP integration for feature selection."""
    print("\n🔧 Testing LGBM-SHAP Integration...")
    
    try:
        # Test if LGBM and SHAP are available
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
        print(f"  ❌ LGBM-SHAP integration test failed: {e}")
        return False

def test_comprehensive_integration_workflow():
    """Test comprehensive integration workflow."""
    print("\n🔧 Testing Comprehensive Integration Workflow...")
    
    try:
        from src.feature_generation.categories.enhanced_hdbscan_clustering_integration import EnhancedHDBSCANClusteringIntegration
        from src.feature_generation.categories.enhanced_regime_clustering_integration import EnhancedRegimeClusteringIntegration
        from src.feature_generation.categories.enhanced_models_training_integration import EnhancedModelsTrainingIntegration
        from src.feature_generation.categories.enhanced_ensemble_training_integration import EnhancedEnsembleTrainingIntegration
        
        # Create sample data
        data = create_sample_data(100)
        
        # Test HDBSCAN clustering workflow
        hdbscan_integrator = EnhancedHDBSCANClusteringIntegration()
        hdbscan_features = hdbscan_integrator.get_comprehensive_clustering_features(data)
        print(f"  ✅ HDBSCAN workflow: {hdbscan_features['feature_count']} features")
        
        # Test regime clustering workflow
        regime_integrator = EnhancedRegimeClusteringIntegration()
        regime_features = regime_integrator.get_comprehensive_regime_features(data)
        print(f"  ✅ Regime clustering workflow: {regime_features['feature_count']} features")
        
        # Test models training workflow
        models_integrator = EnhancedModelsTrainingIntegration()
        models_features = models_integrator.get_comprehensive_training_features(data)
        print(f"  ✅ Models training workflow: {models_features['feature_count']} features")
        
        # Test ensemble training workflow
        ensemble_integrator = EnhancedEnsembleTrainingIntegration()
        ensemble_features = ensemble_integrator.get_comprehensive_ensemble_features(data)
        print(f"  ✅ Ensemble training workflow: {ensemble_features['feature_count']} features")
        
        # Test feature category breakdown
        hdbscan_breakdown = hdbscan_features.get('feature_categories', {})
        regime_breakdown = regime_features.get('feature_categories', {})
        models_breakdown = models_features.get('feature_categories', {})
        ensemble_breakdown = ensemble_features.get('feature_categories', {})
        
        print(f"  ✅ HDBSCAN categories: {hdbscan_breakdown}")
        print(f"  ✅ Regime categories: {regime_breakdown}")
        print(f"  ✅ Models categories: {models_breakdown}")
        print(f"  ✅ Ensemble categories: {ensemble_breakdown}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Comprehensive integration workflow test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Integration System")
    print("=" * 50)
    
    tests = [
        test_feature_bank_integration,
        test_enhanced_hdbscan_integration,
        test_enhanced_regime_clustering_integration,
        test_enhanced_models_training_integration,
        test_enhanced_ensemble_training_integration,
        test_feature_category_breakdown,
        test_lgbm_shap_integration,
        test_comprehensive_integration_workflow
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
        print("\n🎉 All tests passed! Enhanced integration system is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)