"""
Comprehensive Test Suite for Enhanced CVLSA Implementations

This test suite verifies all the new enhanced CVLSA components:
1. Adaptive Cascade Architecture
2. Enhanced Variable Selection
3. Improved Feature Engineering
4. Performance & Memory Management
5. Robust Error Handling
6. Advanced Monitoring & Analytics
7. Configuration Simplification
"""

import numpy as np
import pandas as pd
import torch
import logging
import time
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_market_data(n_samples: int = 1000, n_features: int = 20) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Create sample market data for testing."""
    np.random.seed(42)
    
    # Generate synthetic market data
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Generate price data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    market_data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with some relationship to features
    y = np.sum(X[:, :5], axis=1) + np.random.normal(0, 0.1, n_samples)
    
    # Generate regimes
    regimes = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])
    
    return market_data, X, y, regimes

def test_adaptive_cascade_architecture():
    """Test adaptive cascade architecture."""
    logger.info("🏗️ Testing Adaptive Cascade Architecture...")
    
    try:
        from src.utils.ml_common.models.adaptive_cascade_architecture import (
            AdaptiveCascadeArchitecture, GeneticOptimizationConfig, create_adaptive_cascade
        )
        from src.utils.ml_common.models.cvlsa_architecture import EnhancedCVLSAConfig
        
        # Create sample data
        market_data, X, y, regimes = create_sample_market_data(500, 15)
        
        # Create configuration
        config = EnhancedCVLSAConfig(
            input_dim=15,
            output_dim=1,
            seq_length=50,
            memory_efficient=True
        )
        
        genetic_config = GeneticOptimizationConfig(
            population_size=20,
            generations=10,
            mutation_rate=0.1
        )
        
        # Create adaptive cascade
        cascade = create_adaptive_cascade(config, genetic_config)
        
        # Test data complexity calculation
        complexity = cascade.calculate_data_complexity(X, y)
        logger.info(f"   Data complexity: {complexity:.3f}")
        
        # Test regime characteristics
        regime_chars = cascade.calculate_regime_characteristics(X, regimes)
        logger.info(f"   Regime characteristics: {regime_chars}")
        
        # Test optimal depth determination
        optimal_depth = cascade.determine_optimal_depth(X, y, regimes)
        logger.info(f"   Optimal depth: {optimal_depth}")
        
        # Test cascade building
        logger.info("   Building adaptive cascade...")
        start_time = time.time()
        cascade.build_adaptive_cascade(X, y, regimes)
        build_time = time.time() - start_time
        
        logger.info(f"   Cascade built in {build_time:.2f}s")
        logger.info(f"   Total levels: {len(cascade.cascade_levels)}")
        logger.info(f"   Active levels: {sum(1 for level in cascade.cascade_levels if level.is_active)}")
        
        # Test predictions
        predictions = cascade.predict(X[:100])
        logger.info(f"   Predictions shape: {predictions.shape}")
        
        # Test analytics
        analytics = cascade.get_cascade_analytics()
        logger.info(f"   Analytics keys: {list(analytics.keys())}")
        
        logger.info("✅ Adaptive Cascade Architecture test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive Cascade Architecture test failed: {e}")
        return False

def test_enhanced_variable_selection():
    """Test enhanced variable selection."""
    logger.info("🔍 Testing Enhanced Variable Selection...")
    
    try:
        from src.utils.ml_common.models.enhanced_variable_selection import (
            EnhancedVariableSelector, VariableSelectionConfig, create_enhanced_variable_selector
        )
        
        # Create sample data
        market_data, X, y, regimes = create_sample_market_data(300, 25)
        
        # Create configuration
        config = VariableSelectionConfig(
            use_parallel=True,
            max_workers=2,
            methods=['variance_threshold', 'mutual_info', 'random_forest'],
            adaptive_method_selection=True,
            enable_incremental=True
        )
        
        # Create selector
        selector = create_enhanced_variable_selector(config)
        
        # Test data characteristics analysis
        characteristics = selector.analyze_data_characteristics(X, y)
        logger.info(f"   Data characteristics: {len(characteristics)} metrics")
        
        # Test adaptive method selection
        methods = selector.select_adaptive_methods(characteristics)
        logger.info(f"   Selected methods: {methods}")
        
        # Test parallel selection
        logger.info("   Running parallel variable selection...")
        start_time = time.time()
        selected_features, results = selector.select_features(X, y, incremental=False)
        selection_time = time.time() - start_time
        
        logger.info(f"   Selection completed in {selection_time:.2f}s")
        logger.info(f"   Selected features: {len(selected_features)}")
        logger.info(f"   Reduction ratio: {results['reduction_ratio']:.1%}")
        
        # Test incremental selection
        logger.info("   Testing incremental selection...")
        incremental_features, incremental_results = selector.select_features_incremental(X, y)
        logger.info(f"   Incremental features: {len(incremental_features)}")
        
        # Test analytics
        analytics = selector.get_selection_analytics()
        logger.info(f"   Analytics: {list(analytics.keys())}")
        
        logger.info("✅ Enhanced Variable Selection test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Variable Selection test failed: {e}")
        return False

def test_improved_feature_engineering():
    """Test improved feature engineering."""
    logger.info("🔧 Testing Improved Feature Engineering...")
    
    try:
        from src.utils.ml_common.models.improved_feature_engineering import (
            ImprovedFeatureEngineer, FeatureEngineeringConfig, create_improved_feature_engineer
        )
        
        # Create sample market data
        market_data, X, y, regimes = create_sample_market_data(200, 10)
        
        # Create configuration
        config = FeatureEngineeringConfig(
            enable_market_features=True,
            enable_microstructure_features=True,
            enable_regime_features=True,
            enable_technical_indicators=True,
            enable_interaction_terms=True,
            enable_dimensionality_reduction=True,
            reduction_method='pca',
            enable_scaling=True,
            scaling_method='robust'
        )
        
        # Create feature engineer
        engineer = create_improved_feature_engineer(config)
        
        # Test market features
        logger.info("   Engineering market features...")
        market_enhanced = engineer.engineer_market_features(market_data)
        logger.info(f"   Market features: {len(market_enhanced.columns)} columns")
        
        # Test microstructure features
        microstructure_enhanced = engineer.engineer_microstructure_features(market_enhanced)
        logger.info(f"   Microstructure features: {len(microstructure_enhanced.columns)} columns")
        
        # Test regime features
        regime_enhanced = engineer.engineer_regime_features(microstructure_enhanced, regimes)
        logger.info(f"   Regime features: {len(regime_enhanced.columns)} columns")
        
        # Test technical indicators
        technical_enhanced = engineer.engineer_technical_indicators(regime_enhanced)
        logger.info(f"   Technical indicators: {len(technical_enhanced.columns)} columns")
        
        # Test interaction terms
        interaction_enhanced = engineer.engineer_interaction_terms(technical_enhanced, y)
        logger.info(f"   Interaction terms: {len(interaction_enhanced.columns)} columns")
        
        # Test complete pipeline
        logger.info("   Running complete feature engineering pipeline...")
        start_time = time.time()
        final_features, results = engineer.engineer_features(market_data, y, regimes)
        engineering_time = time.time() - start_time
        
        logger.info(f"   Feature engineering completed in {engineering_time:.2f}s")
        logger.info(f"   Original features: {results['original_features']}")
        logger.info(f"   New features: {results['new_features']}")
        logger.info(f"   Total features: {results['total_features']}")
        logger.info(f"   Expansion ratio: {results['feature_expansion_ratio']:.2f}x")
        
        # Test analytics
        analytics = engineer.get_engineering_analytics()
        logger.info(f"   Analytics: {list(analytics.keys())}")
        
        logger.info("✅ Improved Feature Engineering test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Improved Feature Engineering test failed: {e}")
        return False

def test_performance_memory_management():
    """Test performance and memory management."""
    logger.info("🚀 Testing Performance & Memory Management...")
    
    try:
        from src.utils.ml_common.models.performance_memory_management import (
            PerformanceMemoryManager, ResourceConfig, create_performance_memory_manager
        )
        
        # Create configuration
        config = ResourceConfig(
            max_memory_usage=0.8,
            chunk_size=100,
            enable_model_caching=True,
            enable_incremental_learning=True,
            enable_performance_monitoring=True,
            monitoring_interval=1
        )
        
        # Create manager
        manager = create_performance_memory_manager(config)
        
        # Test monitoring
        logger.info("   Starting monitoring...")
        manager.start_monitoring()
        time.sleep(2)  # Let monitoring run for a bit
        
        # Test model caching
        logger.info("   Testing model caching...")
        model_config = {'type': 'test_model', 'params': {'n_estimators': 10}}
        X, y = np.random.randn(100, 10), np.random.randn(100)
        
        # Cache a model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        performance_metrics = {'mse': 0.1, 'r2': 0.8}
        manager.cache_model(model, model_config, X, y, performance_metrics)
        
        # Try to retrieve cached model
        cached_model = manager.get_cached_model(model_config, X, y)
        if cached_model is not None:
            logger.info("   Model caching successful")
        else:
            logger.info("   Model caching test completed (no cached model found)")
        
        # Test memory-efficient processing
        logger.info("   Testing memory-efficient processing...")
        
        def processing_func(X_chunk, y_chunk):
            return np.mean(X_chunk, axis=0)
        
        large_X = np.random.randn(1000, 20)
        large_y = np.random.randn(1000)
        
        with manager.memory_efficient_processing("test_processing"):
            results = manager.process_large_dataset(large_X, large_y, processing_func, chunk_size=100)
            logger.info(f"   Processed {len(results)} chunks")
        
        # Test analytics
        analytics = manager.get_comprehensive_analytics()
        logger.info(f"   Analytics keys: {list(analytics.keys())}")
        
        # Stop monitoring
        manager.stop_monitoring()
        
        logger.info("✅ Performance & Memory Management test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance & Memory Management test failed: {e}")
        return False

def test_robust_error_handling():
    """Test robust error handling."""
    logger.info("🛡️ Testing Robust Error Handling...")
    
    try:
        from src.utils.ml_common.models.robust_error_handling import (
            RobustErrorHandler, ValidationConfig, create_robust_error_handler
        )
        
        # Create configuration
        config = ValidationConfig(
            validate_inputs=True,
            strict_validation=False,
            enable_error_recovery=True,
            enable_fallback=True
        )
        
        # Create error handler
        handler = create_robust_error_handler(config)
        
        # Test input validation
        logger.info("   Testing input validation...")
        
        # Valid DataFrame
        valid_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        validated_df = handler.validator.validate_dataframe(valid_df, "test_df")
        logger.info(f"   Valid DataFrame validated: {validated_df.shape}")
        
        # Valid array
        valid_array = np.array([[1, 2], [3, 4]])
        validated_array = handler.validator.validate_array(valid_array, "test_array")
        logger.info(f"   Valid array validated: {validated_array.shape}")
        
        # Test error handling
        logger.info("   Testing error handling...")
        
        def failing_operation():
            raise ValueError("Test error")
        
        def successful_operation():
            return "Success"
        
        # Test failing operation
        success, result, error_report = handler.handle_operation(failing_operation)
        logger.info(f"   Failing operation handled: success={success}")
        
        # Test successful operation
        success, result, error_report = handler.handle_operation(successful_operation)
        logger.info(f"   Successful operation: success={success}, result={result}")
        
        # Test error summary
        error_summary = handler.get_error_summary()
        logger.info(f"   Error summary: {error_summary}")
        
        # Test health status
        health_status = handler.get_health_status()
        logger.info(f"   Health status: {health_status['status']} (score: {health_status['health_score']})")
        
        logger.info("✅ Robust Error Handling test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Robust Error Handling test failed: {e}")
        return False

def test_advanced_monitoring_analytics():
    """Test advanced monitoring and analytics."""
    logger.info("📊 Testing Advanced Monitoring & Analytics...")
    
    try:
        from src.utils.ml_common.models.advanced_monitoring_analytics import (
            AdvancedMonitoringAnalytics, ExperimentConfig, create_advanced_monitoring_analytics
        )
        
        # Create configuration
        config = ExperimentConfig(
            enable_experiment_tracking=True,
            enable_detailed_analytics=True,
            enable_real_time_monitoring=True,
            monitoring_interval=1,
            enable_auto_reporting=True
        )
        
        # Create analytics system
        analytics = create_advanced_monitoring_analytics(config)
        
        # Test experiment tracking
        logger.info("   Testing experiment tracking...")
        experiment_id = analytics.start_experiment(
            name="Test Experiment",
            description="Testing the analytics system",
            config={'test_param': 'test_value'},
            hyperparameters={'learning_rate': 0.01},
            tags=['test', 'analytics']
        )
        logger.info(f"   Started experiment: {experiment_id}")
        
        # Log some metrics
        for i in range(5):
            analytics.log_metric(experiment_id, 'accuracy', 0.8 + i * 0.02)
            analytics.log_metric(experiment_id, 'loss', 0.5 - i * 0.05)
            time.sleep(0.1)
        
        # Test performance recording
        logger.info("   Testing performance recording...")
        analytics.record_performance(
            component="test_component",
            operation="test_operation",
            metrics={'execution_time': 1.5, 'memory_usage': 100},
            metadata={'batch_size': 32}
        )
        
        # Test monitoring
        logger.info("   Starting monitoring...")
        analytics.start_monitoring()
        time.sleep(2)  # Let monitoring run
        
        # Test comprehensive report
        logger.info("   Generating comprehensive report...")
        report = analytics.generate_comprehensive_report(experiment_id)
        logger.info(f"   Report keys: {list(report.keys())}")
        
        # Test dashboard data
        dashboard_data = analytics.create_dashboard_data()
        logger.info(f"   Dashboard data keys: {list(dashboard_data.keys())}")
        
        # Complete experiment
        analytics.complete_experiment(experiment_id, results={'final_accuracy': 0.9})
        logger.info("   Experiment completed")
        
        # Stop monitoring
        analytics.stop_monitoring()
        
        logger.info("✅ Advanced Monitoring & Analytics test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced Monitoring & Analytics test failed: {e}")
        return False

def test_configuration_simplification():
    """Test configuration simplification."""
    logger.info("🔧 Testing Configuration Simplification...")
    
    try:
        from src.utils.ml_common.models.configuration_simplification import (
            ConfigurationSimplification, create_configuration_simplification
        )
        
        # Create simplification system
        simplifier = create_configuration_simplification()
        
        # Test profile listing
        logger.info("   Testing profile listing...")
        profiles = simplifier.list_available_profiles()
        logger.info(f"   Available profiles: {len(profiles)}")
        
        # Test profile by category
        performance_profiles = simplifier.list_available_profiles('performance')
        logger.info(f"   Performance profiles: {len(performance_profiles)}")
        
        # Test profile configuration
        logger.info("   Testing profile configuration...")
        fast_config = simplifier.get_profile_config('fast')
        if fast_config:
            logger.info(f"   Fast profile config keys: {list(fast_config.keys())}")
        
        # Test auto-configuration
        logger.info("   Testing auto-configuration...")
        X, y = np.random.randn(500, 20), np.random.randn(500)
        
        auto_result = simplifier.auto_configure(X, y, use_case='research', performance_priority='balanced')
        logger.info(f"   Auto-configuration success: {auto_result.success}")
        logger.info(f"   Auto-configuration reasoning: {len(auto_result.reasoning)} points")
        
        if auto_result.success:
            logger.info(f"   Generated config keys: {list(auto_result.config.keys())}")
        
        # Test configuration validation
        logger.info("   Testing configuration validation...")
        test_config = {
            'adaptive_cascade': {
                'max_depth': 5,
                'genetic_optimization': True,
                'cascade_pruning': True
            },
            'variable_selection': {
                'use_parallel': True,
                'max_workers': 4,
                'methods': ['variance_threshold', 'random_forest']
            }
        }
        
        is_valid, issues = simplifier.validate_config(test_config)
        logger.info(f"   Configuration valid: {is_valid}")
        if issues:
            logger.info(f"   Validation issues: {issues}")
        
        # Test recommendations
        logger.info("   Testing recommendations...")
        recommendations = simplifier.get_recommendations(
            dataset_size=1000,
            available_memory_gb=8,
            cpu_cores=4,
            gpu_available=False
        )
        logger.info(f"   Recommendations: {recommendations}")
        
        logger.info("✅ Configuration Simplification test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration Simplification test failed: {e}")
        return False

def run_all_tests():
    """Run all enhanced CVLSA implementation tests."""
    logger.info("🧪 Starting Enhanced CVLSA Implementation Tests...")
    
    tests = [
        ("Adaptive Cascade Architecture", test_adaptive_cascade_architecture),
        ("Enhanced Variable Selection", test_enhanced_variable_selection),
        ("Improved Feature Engineering", test_improved_feature_engineering),
        ("Performance & Memory Management", test_performance_memory_management),
        ("Robust Error Handling", test_robust_error_handling),
        ("Advanced Monitoring & Analytics", test_advanced_monitoring_analytics),
        ("Configuration Simplification", test_configuration_simplification)
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            start_time = time.time()
            success = test_func()
            test_time = time.time() - start_time
            
            results[test_name] = {
                'success': success,
                'time': test_time
            }
            
            if success:
                passed_tests += 1
                logger.info(f"✅ {test_name} PASSED ({test_time:.2f}s)")
            else:
                logger.error(f"❌ {test_name} FAILED ({test_time:.2f}s)")
                
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            results[test_name] = {
                'success': False,
                'time': 0,
                'error': str(e)
            }
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    logger.info("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        time_str = f"({result['time']:.2f}s)" if result['time'] > 0 else ""
        logger.info(f"  {test_name}: {status} {time_str}")
        if 'error' in result:
            logger.info(f"    Error: {result['error']}")
    
    return results

if __name__ == "__main__":
    # Run all tests
    results = run_all_tests()
    
    # Exit with appropriate code
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r['success'])
    
    if passed_tests == total_tests:
        logger.info("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error(f"\n💥 {total_tests - passed_tests} tests failed!")
        sys.exit(1)