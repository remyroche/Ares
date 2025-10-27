"""
Minimal Feature Task Integration Test

This script tests only the core integration functionality without complex dependencies.
"""

import numpy as np
import pandas as pd
import warnings
import sys
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_data(n_samples: int = 100) -> pd.DataFrame:
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


def test_basic_feature_categorization():
    """Test basic feature categorization without complex dependencies."""
    print("\n🔧 Testing Basic Feature Categorization...")
    
    try:
        # Test the basic categorization structure
        from src.feature_generation.categories.regime_feature_categorization import (
            FeatureUseCase, get_hdbscan_features, get_regime_clustering_features,
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
        
        # Check if feature counts are within expected ranges
        assert 50 <= len(hdbscan_features) <= 100, f"HDBSCAN features count {len(hdbscan_features)} not in range [50, 100]"
        assert 40 <= len(regime_features) <= 80, f"Regime clustering features count {len(regime_features)} not in range [40, 80]"
        assert 30 <= len(training_features) <= 60, f"Models training features count {len(training_features)} not in range [30, 60]"
        assert 20 <= len(ensemble_features) <= 40, f"Ensemble training features count {len(ensemble_features)} not in range [20, 40]"
        
        print("  ✅ All feature counts are within expected ranges")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic feature categorization test failed: {e}")
        return False


def test_feature_task_integration_basic():
    """Test basic feature task integration."""
    print("\n🔧 Testing Basic Feature Task Integration...")
    
    try:
        from src.feature_generation.categories.feature_task_integration import (
            FeatureTaskIntegrator, MLTask
        )
        
        # Create sample data
        data = create_sample_data(50)
        
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
        
        return results
        
    except Exception as e:
        print(f"  ❌ Basic feature task integration test failed: {e}")
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
        y = np.random.randint(0, 2, n_samples)
        
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
        from src.feature_generation.categories.ensemble_training_integration import EnsembleTrainingIntegration
        
        # Create sample data
        data = create_sample_data(50)
        
        # Test ensemble integration
        integrator = EnsembleTrainingIntegration()
        feature_result = integrator.get_ensemble_features(data)
        
        print(f"  ✅ Generated {feature_result['feature_count']} ensemble features")
        print(f"  ✅ Includes base outputs: {feature_result['includes_base_outputs']}")
        print(f"  ✅ Includes disagreement: {feature_result['includes_disagreement']}")
        print(f"  ✅ Includes entropy: {feature_result['includes_entropy']}")
        
        # Test synthetic target creation
        target = integrator._create_synthetic_ensemble_target(data)
        print(f"  ✅ Synthetic target created: {target.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic ensemble features test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Minimal Feature Task Integration Tests")
    print("=" * 60)
    
    # Test results
    test_results = {}
    
    # Run tests
    test_results['basic_feature_categorization'] = test_basic_feature_categorization()
    test_results['feature_task_integration_basic'] = test_feature_task_integration_basic()
    test_results['lgbm_shap_basic'] = test_lgbm_shap_basic()
    test_results['ensemble_features_basic'] = test_ensemble_features_basic()
    
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