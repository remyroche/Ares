"""
Test Feature Task Integration System

This script tests the integration between distinct feature categories and their respective ML tasks.
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import the integration modules
try:
    from src.feature_generation.categories.feature_task_integration import (
        FeatureTaskIntegrator, MLTask, validate_all_feature_mappings
    )
    from src.feature_generation.categories.hdbscan_clustering_integration import (
        HDBSCANClusteringIntegration, get_hdbscan_clustering_features
    )
    from src.feature_generation.categories.regime_clustering_integration import (
        RegimeClusteringIntegration, get_regime_clustering_features
    )
    from src.feature_generation.categories.models_training_integration import (
        ModelsTrainingIntegration, get_models_training_features
    )
    from src.feature_generation.categories.ensemble_training_integration import (
        EnsembleTrainingIntegration, get_ensemble_training_features
    )
    print("✅ Successfully imported all integration modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


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


def test_feature_task_integrator():
    """Test the main feature task integrator."""
    print("\n🔧 Testing Feature Task Integrator...")
    
    # Create sample data
    data = create_sample_data(500)
    
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


def test_hdbscan_clustering_integration():
    """Test HDBSCAN clustering integration."""
    print("\n🔧 Testing HDBSCAN Clustering Integration...")
    
    # Create sample data
    data = create_sample_data(300)
    
    try:
        # Test feature generation
        integrator = HDBSCANClusteringIntegration()
        feature_result = integrator.get_clustering_features(data)
        
        print(f"  ✅ Generated {feature_result['feature_count']} clustering features")
        print(f"  ✅ Target range: {feature_result['target_range']}")
        print(f"  ✅ Clustering optimized: {feature_result['clustering_optimized']}")
        
        # Test data preparation
        feature_matrix, feature_names = integrator.prepare_data_for_clustering(data)
        print(f"  ✅ Prepared feature matrix: {feature_matrix.shape}")
        print(f"  ✅ Feature names: {len(feature_names)}")
        
        # Test clustering (if HDBSCAN is available)
        try:
            clustering_result = integrator.cluster_with_hdbscan(data)
            print(f"  ✅ Clustering completed: {clustering_result['n_clusters']} clusters")
            print(f"  ✅ Noise points: {clustering_result['n_noise']}")
            
            # Test quality analysis
            quality_analysis = integrator.analyze_clustering_quality(clustering_result)
            print(f"  ✅ Clustering quality: {quality_analysis['clustering_quality']}")
            
        except ImportError:
            print("  ⚠️ HDBSCAN not available, skipping clustering test")
        
        return True
        
    except Exception as e:
        print(f"  ❌ HDBSCAN clustering integration failed: {e}")
        return False


def test_regime_clustering_integration():
    """Test regime clustering integration."""
    print("\n🔧 Testing Regime Clustering Integration...")
    
    # Create sample data
    data = create_sample_data(300)
    
    try:
        # Test feature generation
        integrator = RegimeClusteringIntegration()
        feature_result = integrator.get_regime_features(data)
        
        print(f"  ✅ Generated {feature_result['feature_count']} regime features")
        print(f"  ✅ Target range: {feature_result['target_range']}")
        print(f"  ✅ Regime optimized: {feature_result['regime_optimized']}")
        
        # Test data preparation
        feature_matrix, feature_names = integrator.prepare_data_for_clustering(data)
        print(f"  ✅ Prepared feature matrix: {feature_matrix.shape}")
        print(f"  ✅ Feature names: {len(feature_names)}")
        
        # Test clustering
        try:
            clustering_result = integrator.cluster_regimes(data, n_clusters=3)
            print(f"  ✅ Regime clustering completed: {clustering_result['n_clusters']} clusters")
            print(f"  ✅ Algorithm: {clustering_result['algorithm']}")
            
            # Test regime analysis
            regime_analysis = integrator.analyze_regime_characteristics(data, clustering_result)
            print(f"  ✅ Regime analysis: {regime_analysis['n_regimes']} regimes identified")
            
        except ImportError:
            print("  ⚠️ Scikit-learn not available, skipping clustering test")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Regime clustering integration failed: {e}")
        return False


def test_models_training_integration():
    """Test models training integration."""
    print("\n🔧 Testing Models Training Integration...")
    
    # Create sample data
    data = create_sample_data(300)
    
    try:
        # Test feature generation
        integrator = ModelsTrainingIntegration()
        feature_result = integrator.get_training_features(data)
        
        print(f"  ✅ Generated {feature_result['feature_count']} training features")
        print(f"  ✅ Target range: {feature_result['target_range']}")
        print(f"  ✅ Training optimized: {feature_result['training_optimized']}")
        print(f"  ✅ Selection method: {feature_result['selection_method']}")
        
        # Test data preparation
        target = integrator.create_synthetic_target(data, 'regime')
        feature_matrix, feature_names, target = integrator.prepare_data_for_training(data, target)
        print(f"  ✅ Prepared feature matrix: {feature_matrix.shape}")
        print(f"  ✅ Target shape: {target.shape}")
        
        # Test model training
        try:
            training_result = integrator.train_model_with_features(data, target, 'lgbm')
            print(f"  ✅ Model training completed: {training_result['model_type']}")
            print(f"  ✅ R² score: {training_result['r2']:.4f}")
            print(f"  ✅ MSE: {training_result['mse']:.4f}")
            
            # Test feature importance analysis
            importance_analysis = integrator.analyze_feature_importance(training_result)
            print(f"  ✅ Feature importance analysis: {len(importance_analysis['top_features'])} top features")
            
        except ImportError:
            print("  ⚠️ LGBM not available, skipping model training test")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Models training integration failed: {e}")
        return False


def test_ensemble_training_integration():
    """Test ensemble training integration."""
    print("\n🔧 Testing Ensemble Training Integration...")
    
    # Create sample data
    data = create_sample_data(300)
    
    try:
        # Test feature generation
        integrator = EnsembleTrainingIntegration()
        feature_result = integrator.get_ensemble_features(data)
        
        print(f"  ✅ Generated {feature_result['feature_count']} ensemble features")
        print(f"  ✅ Target range: {feature_result['target_range']}")
        print(f"  ✅ Ensemble optimized: {feature_result['ensemble_optimized']}")
        print(f"  ✅ Includes base outputs: {feature_result['includes_base_outputs']}")
        print(f"  ✅ Includes disagreement: {feature_result['includes_disagreement']}")
        print(f"  ✅ Includes entropy: {feature_result['includes_entropy']}")
        
        # Test data preparation
        target = integrator._create_synthetic_ensemble_target(data)
        feature_matrix, feature_names, target = integrator.prepare_data_for_ensemble_training(data, target)
        print(f"  ✅ Prepared feature matrix: {feature_matrix.shape}")
        print(f"  ✅ Target shape: {target.shape}")
        
        # Test ensemble training
        try:
            training_result = integrator.train_ensemble_meta_learner(data, target, meta_learner_type='linear')
            print(f"  ✅ Ensemble training completed: {training_result['meta_learner_type']}")
            print(f"  ✅ R² score: {training_result['r2']:.4f}")
            print(f"  ✅ MSE: {training_result['mse']:.4f}")
            
            # Test performance analysis
            performance_analysis = integrator.analyze_ensemble_performance(training_result)
            print(f"  ✅ Performance analysis: RMSE = {performance_analysis['rmse']:.4f}")
            
        except ImportError:
            print("  ⚠️ Scikit-learn not available, skipping ensemble training test")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ensemble training integration failed: {e}")
        return False


def test_feature_validation():
    """Test feature validation across all tasks."""
    print("\n🔧 Testing Feature Validation...")
    
    try:
        validation_results = validate_all_feature_mappings()
        
        print("  📊 Feature Validation Results:")
        for task, result in validation_results.items():
            status = "✅" if result['within_range'] else "⚠️"
            print(f"    {status} {task}: {result['feature_count']} features ({result['description']})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature validation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Feature Task Integration Tests")
    print("=" * 50)
    
    # Test results
    test_results = {}
    
    # Run tests
    test_results['feature_task_integrator'] = test_feature_task_integrator()
    test_results['hdbscan_clustering'] = test_hdbscan_clustering_integration()
    test_results['regime_clustering'] = test_regime_clustering_integration()
    test_results['models_training'] = test_models_training_integration()
    test_results['ensemble_training'] = test_ensemble_training_integration()
    test_results['feature_validation'] = test_feature_validation()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
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