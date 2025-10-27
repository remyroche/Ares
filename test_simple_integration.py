"""
Simple Feature Task Integration Test

This script tests the core integration functionality without complex dependencies.
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='15T')
    
    # Generate realistic price data
    np.random.seed(42)
    price = 100
    prices = [price]
    volumes = []
    
    for i in range(n_samples - 1):
        # Generate price movement
        change = np.random.normal(0, 0.001)
        price *= (1 + change)
        prices.append(price)
        
        # Generate volume
        volume = np.random.lognormal(10, 0.5)
        volumes.append(volume)
    
    # Add final volume
    volumes.append(np.random.lognormal(10, 0.5))
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    return data


def test_feature_categorization():
    """Test the feature categorization system."""
    print("\n🔧 Testing Feature Categorization...")
    
    try:
        # Import the categorization module
        from src.feature_generation.categories.regime_feature_categorization import (
            RegimeFeatureCategorizer, FeatureUseCase, 
            get_hdbscan_features, get_regime_clustering_features,
            get_regime_models_training_features, get_regime_ensemble_training_features
        )
        
        # Test feature counts
        hdbscan_features = get_hdbscan_features()
        regime_features = get_regime_clustering_features()
        training_features = get_regime_models_training_features()
        ensemble_features = get_regime_ensemble_training_features()
        
        print(f"  ✅ HDBSCAN features: {len(hdbscan_features)}")
        print(f"  ✅ Regime clustering features: {len(regime_features)}")
        print(f"  ✅ Models training features: {len(training_features)}")
        print(f"  ✅ Ensemble training features: {len(ensemble_features)}")
        
        # Test categorizer
        categorizer = RegimeFeatureCategorizer()
        
        # Test feature requirements
        for use_case in FeatureUseCase:
            requirements = categorizer.get_feature_requirements(use_case)
            print(f"  ✅ {use_case.value}: {requirements['total_features']} features, {requirements['total_categories']} categories")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature categorization test failed: {e}")
        return False


def test_feature_task_integration():
    """Test the feature task integration system."""
    print("\n🔧 Testing Feature Task Integration...")
    
    try:
        # Import the integration module
        from src.feature_generation.categories.feature_task_integration import (
            FeatureTaskIntegrator, MLTask, validate_all_feature_mappings
        )
        
        # Create sample data
        data = create_sample_data(200)
        
        # Initialize integrator
        integrator = FeatureTaskIntegrator()
        
        # Test each task
        tasks = [
            MLTask.HDBSCAN_CLUSTERING,
            MLTask.REGIME_CLUSTERING,
            MLTask.REGIME_MODELS_TRAINING,
            MLTask.REGIME_ENSEMBLE_TRAINING
        ]
        
        results = {}
        for task in tasks:
            try:
                result = integrator.get_features_for_task(task, data)
                results[task.value] = {
                    'success': True,
                    'feature_count': result['feature_count'],
                    'within_range': result['target_range'][0] <= result['feature_count'] <= result['target_range'][1],
                    'description': result['description']
                }
                print(f"  ✅ {task.value}: {result['feature_count']} features")
            except Exception as e:
                results[task.value] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"  ❌ {task.value}: {e}")
        
        # Test validation
        validation_results = validate_all_feature_mappings()
        print(f"  ✅ Validation completed for {len(validation_results)} tasks")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Feature task integration test failed: {e}")
        return False


def test_lgbm_shap_selection():
    """Test LGBM-SHAP feature selection."""
    print("\n🔧 Testing LGBM-SHAP Feature Selection...")
    
    try:
        # Import required modules
        import lightgbm as lgb
        import shap
        
        # Create sample data
        data = create_sample_data(200)
        
        # Create synthetic features
        n_features = 100
        feature_names = [f'feature_{i}' for i in range(n_features)]
        feature_data = {}
        
        for i, name in enumerate(feature_names):
            feature_data[name] = np.random.randn(len(data))
        
        # Create synthetic target
        target = np.random.randint(0, 2, len(data))
        
        # Test LGBM-SHAP selection
        from src.feature_generation.categories.models_training_integration import ModelsTrainingIntegration
        
        integrator = ModelsTrainingIntegration()
        selected_features = integrator._select_features_with_lgbm_shap(
            data, feature_data, target, 30
        )
        
        print(f"  ✅ Selected {len(selected_features)} features from {n_features}")
        print(f"  ✅ Selection ratio: {len(selected_features)/n_features:.2%}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ LGBM-SHAP selection test failed: {e}")
        return False


def test_ensemble_features():
    """Test ensemble feature generation."""
    print("\n🔧 Testing Ensemble Feature Generation...")
    
    try:
        from src.feature_generation.categories.ensemble_training_integration import EnsembleTrainingIntegration
        
        # Create sample data
        data = create_sample_data(200)
        
        # Test ensemble integration
        integrator = EnsembleTrainingIntegration()
        feature_result = integrator.get_ensemble_features(data)
        
        print(f"  ✅ Generated {feature_result['feature_count']} ensemble features")
        print(f"  ✅ Includes base outputs: {feature_result['includes_base_outputs']}")
        print(f"  ✅ Includes disagreement: {feature_result['includes_disagreement']}")
        print(f"  ✅ Includes entropy: {feature_result['includes_entropy']}")
        
        # Test data preparation
        target = integrator._create_synthetic_ensemble_target(data)
        feature_matrix, feature_names, target = integrator.prepare_data_for_ensemble_training(data, target)
        
        print(f"  ✅ Prepared feature matrix: {feature_matrix.shape}")
        print(f"  ✅ Target shape: {target.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ensemble feature test failed: {e}")
        return False


def test_clustering_integration():
    """Test clustering integration."""
    print("\n🔧 Testing Clustering Integration...")
    
    try:
        from src.feature_generation.categories.hdbscan_clustering_integration import HDBSCANClusteringIntegration
        from src.feature_generation.categories.regime_clustering_integration import RegimeClusteringIntegration
        
        # Create sample data
        data = create_sample_data(200)
        
        # Test HDBSCAN integration
        hdbscan_integrator = HDBSCANClusteringIntegration()
        hdbscan_result = hdbscan_integrator.get_clustering_features(data)
        print(f"  ✅ HDBSCAN features: {hdbscan_result['feature_count']}")
        
        # Test regime clustering integration
        regime_integrator = RegimeClusteringIntegration()
        regime_result = regime_integrator.get_regime_features(data)
        print(f"  ✅ Regime clustering features: {regime_result['feature_count']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Clustering integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Simple Feature Task Integration Tests")
    print("=" * 60)
    
    # Test results
    test_results = {}
    
    # Run tests
    test_results['feature_categorization'] = test_feature_categorization()
    test_results['feature_task_integration'] = test_feature_task_integration()
    test_results['lgbm_shap_selection'] = test_lgbm_shap_selection()
    test_results['ensemble_features'] = test_ensemble_features()
    test_results['clustering_integration'] = test_clustering_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        if isinstance(result, dict):
            # Feature task integrator results
            success_count = sum(1 for r in result.values() if r.get('success', False))
            total_count = len(result)
            status = f"✅ {success_count}/{total_count}" if success_count == total_count else f"⚠️ {success_count}/{total_count}"
        else:
            # Boolean results
            status = "✅ PASSED" if result else "❌ FAILED"
            if result:
                passed_tests += 1
        
        print(f"  {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Feature task integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()