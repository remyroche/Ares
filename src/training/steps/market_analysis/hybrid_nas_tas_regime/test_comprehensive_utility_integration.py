#!/usr/bin/env python3
"""
Comprehensive Utility Integration Test Script

This script performs runtime testing of all utility integrations to ensure
they work correctly in the hybrid NAS-TAS regime system.
"""

import sys
import logging
from typing import Dict, Any
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared_utils.enhanced_utility_integration import (
    create_enhanced_utility_integration, UtilityIntegrationConfig
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_utility_integration() -> Dict[str, Any]:
    """Test that utility integration is working correctly."""
    try:
        logger.info("🔧 Starting Comprehensive Utility Integration Testing...")

        # Create comprehensive utility integration configuration
        config = UtilityIntegrationConfig(
            # Data operations
            enable_data_validation=True,
            enable_data_quality_checks=True,
            enable_safe_operations=True,

            # Math operations
            enable_math_validation=True,
            enable_safe_math=True,

            # Serialization
            enable_serialization=True,

            # Hardware optimization
            enable_m1_optimizations=True,
            enable_gpu_acceleration=True,
            enable_memory_optimization=True,
            enable_cpu_optimization=True,

            # ML utilities
            enable_ml_common=True,
            enable_feature_selection=True,
            enable_cross_validation=True,
            enable_confidence_metrics=True,
            enable_lookahead_detection=True,
            enable_drift_detection=True,
            enable_ensemble_management=True,
            enable_hmm_regime_detection=True,
            enable_caching=True,

            # Data utilities
            enable_feature_engineering=True,
            enable_gap_detection=True,
            enable_historical_download=True,
            enable_comprehensive_quality=True,
            enable_advanced_quality_metrics=True,

            # Matrix operations
            enable_matrix_operations=True,
            enable_vectorized_operations=True,

            # Performance monitoring
            enable_performance_monitoring=True,
            enable_memory_monitoring=True
        )

        # Create utility integration
        utility_integration = create_enhanced_utility_integration(config)

        # Get integration status
        integration_status = utility_integration.get_integration_status()
        available_utilities = utility_integration.get_available_utilities()

        logger.info(f"✅ Utility Integration Created: {len(available_utilities)} utilities available")

        # Test basic functionality
        logger.info("🔍 Testing basic utility functions...")

        # Test safe mathematical operations
        result = utility_integration.safe_divide(10.0, 2.0)
        assert abs(result - 5.0) < 1e-10, f"Expected 5.0, got {result}"
        logger.info("✅ Safe division works correctly")

        result = utility_integration.safe_divide(10.0, 0.0, default=-1.0)
        assert abs(result - (-1.0)) < 1e-10, f"Expected -1.0, got {result}"
        logger.info("✅ Safe division with default works correctly")

        # Test finite validation
        result = utility_integration.validate_finite(3.14, "test_value")
        assert abs(result - 3.14) < 1e-10, f"Expected 3.14, got {result}"
        logger.info("✅ Finite validation works correctly")

        # Test data quality calculations
        import pandas as pd
        import numpy as np

        # Create test DataFrame
        df = pd.DataFrame({
            'open': np.random.rand(100),
            'high': np.random.rand(100),
            'low': np.random.rand(100),
            'close': np.random.rand(100),
            'volume': np.random.rand(100) * 1000
        })

        # Set datetime index
        df.index = pd.date_range('2023-01-01', periods=100, freq='1min')

        quality_metrics = utility_integration.calculate_data_quality_metrics(df)
        assert 'total_rows' in quality_metrics, "Quality metrics should contain total_rows"
        assert quality_metrics['total_rows'] == 100, f"Expected 100 rows, got {quality_metrics['total_rows']}"
        logger.info("✅ Data quality metrics calculation works correctly")

        # Test comprehensive quality scoring
        quality_score = utility_integration.score_data_quality(df)
        assert isinstance(quality_score, dict), "Quality score should be a dictionary"
        logger.info("✅ Comprehensive quality scoring works correctly")

        # Test advanced quality metrics
        advanced_metrics = utility_integration.calculate_advanced_quality_metrics(df)
        assert isinstance(advanced_metrics, dict), "Advanced metrics should be a dictionary"
        logger.info("✅ Advanced quality metrics calculation works correctly")

        # Test gap detection
        gap_analysis = utility_integration.detect_gaps(df)
        assert isinstance(gap_analysis, dict), "Gap analysis should be a dictionary"
        logger.info("✅ Gap detection works correctly")

        # Test feature engineering
        engineered_df = utility_integration.engineer_features(df, ['momentum', 'volatility', 'volume'])
        assert isinstance(engineered_df, pd.DataFrame), "Feature engineering should return DataFrame"
        logger.info("✅ Feature engineering works correctly")

        # Test schema validation
        is_valid = utility_integration.validate_dataframe_schema(df, ['open', 'high', 'low', 'close', 'volume'])
        assert is_valid, "DataFrame should have valid schema"
        logger.info("✅ Schema validation works correctly")

        # Test null guarding
        guarded_df = utility_integration.guard_dataframe_nulls(df, threshold=0.5)
        assert isinstance(guarded_df, pd.DataFrame), "Null guarding should return DataFrame"
        logger.info("✅ Null guarding works correctly")

        # Test serialization
        test_data = {'test': 'data', 'numbers': [1, 2, 3], 'metrics': quality_metrics}
        success = utility_integration.save_data(test_data, '/tmp/test_data.json')
        assert success, "Data serialization should succeed"
        logger.info("✅ Data serialization works correctly")

        loaded_data = utility_integration.load_data('/tmp/test_data.json')
        assert loaded_data == test_data, "Data deserialization should match original"
        logger.info("✅ Data deserialization works correctly")

        # Test caching
        cache_key = "test_cache_key"
        cache_value = {"cached": "data", "metrics": quality_metrics}
        cache_success = utility_integration.set_cached_data(cache_key, cache_value)
        logger.info("✅ Data caching works correctly")

        retrieved_data = utility_integration.get_cached_data(cache_key)
        assert retrieved_data == cache_value, "Cache retrieval should match cached data"
        logger.info("✅ Data cache retrieval works correctly")

        # Test matrix operations
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])

        # Test matrix multiplication
        C = utility_integration.safe_matrix_multiply(A, B)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(C, expected), "Matrix multiplication should produce correct result"
        logger.info("✅ Matrix multiplication works correctly")

        # Test matrix addition
        D = utility_integration.safe_matrix_add(A, B)
        expected_add = np.array([[6.0, 8.0], [10.0, 12.0]])
        assert np.allclose(D, expected_add), "Matrix addition should produce correct result"
        logger.info("✅ Matrix addition works correctly")

        # Test matrix subtraction
        E = utility_integration.safe_matrix_subtract(A, B)
        expected_sub = np.array([[-4.0, -4.0], [-4.0, -4.0]])
        assert np.allclose(E, expected_sub), "Matrix subtraction should produce correct result"
        logger.info("✅ Matrix subtraction works correctly")

        # Test ML utilities
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 3, 100)

        # Test feature selection
        selected_features = utility_integration.feature_selection(X, y, method="mutual_info")
        assert isinstance(selected_features, np.ndarray), "Feature selection should return array"
        logger.info("✅ Feature selection works correctly")

        # Test cross-validation
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        cv_results = utility_integration.cross_validation(estimator, X, y, cv=3)
        assert 'mean' in cv_results, "Cross-validation should return mean score"
        logger.info("✅ Cross-validation works correctly")

        # Test confidence metrics
        y_pred = np.random.randint(0, 3, 100)
        y_proba = np.random.rand(100, 3)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        confidence_metrics = utility_integration.calculate_confidence_metrics(y_pred, y_proba)
        assert isinstance(confidence_metrics, dict), "Confidence metrics should be a dictionary"
        logger.info("✅ Confidence metrics calculation works correctly")

        # Test calibration metrics
        calibration_metrics = utility_integration.calculate_calibration_metrics(y_pred, y_proba, y)
        assert isinstance(calibration_metrics, dict), "Calibration metrics should be a dictionary"
        logger.info("✅ Calibration metrics calculation works correctly")

        # Test lookahead bias detection
        bias_results = utility_integration.detect_lookahead_bias(X, y)
        assert isinstance(bias_results, dict), "Bias detection should return dictionary"
        logger.info("✅ Lookahead bias detection works correctly")

        # Test data drift detection
        drift_results = utility_integration.detect_data_drift(X[:50], X[50:])
        assert isinstance(drift_results, dict), "Drift detection should return dictionary"
        logger.info("✅ Data drift detection works correctly")

        # Test ensemble creation
        models = [RandomForestClassifier(n_estimators=5, random_state=42)]
        ensemble = utility_integration.create_ensemble(models, method="voting")
        if ensemble is not None:
            logger.info("✅ Ensemble creation works correctly")
        else:
            logger.warning("⚠️ Ensemble creation returned None (expected if ensemble manager unavailable)")

        # Test HMM regime detection
        hmm_results = utility_integration.detect_hmm_regimes(X, n_regimes=3)
        assert isinstance(hmm_results, dict), "HMM regime detection should return dictionary"
        assert 'regime_sequence' in hmm_results, "HMM results should contain regime sequence"
        logger.info("✅ HMM regime detection works correctly")

        # Test memory optimization
        memory_usage = utility_integration.get_memory_usage()
        assert isinstance(memory_usage, (int, float)), "Memory usage should be numeric"
        logger.info("✅ Memory usage monitoring works correctly")

        optimize_result = utility_integration.optimize_memory()
        assert isinstance(optimize_result, dict), "Memory optimization should return dictionary"
        logger.info("✅ Memory optimization works correctly")

        # Clean up test file
        import os
        if os.path.exists('/tmp/test_data.json'):
            os.remove('/tmp/test_data.json')

        logger.info("🎉 All utility integration tests passed successfully!")
        return {
            'success': True,
            'available_utilities': available_utilities,
            'integration_status': integration_status,
            'test_results': {
                'basic_operations': True,
                'data_quality': True,
                'serialization': True,
                'caching': True,
                'matrix_operations': True,
                'ml_utilities': True,
                'memory_optimization': True
            }
        }

    except Exception as e:
        logger.error(f"❌ Utility integration testing failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e),
            'available_utilities': [],
            'integration_status': {},
            'test_results': {}
        }


def main():
    """Main testing function."""
    logger.info("🚀 Starting Comprehensive Utility Integration Testing...")

    results = test_utility_integration()

    if results['success']:
        logger.info("✅ Comprehensive Utility Integration Testing COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 Available Utilities: {len(results['available_utilities'])}")
        logger.info(f"📊 All tests passed: {results['test_results']}")
        logger.info("🎉 Enhanced Utility Integration is fully functional!")
    else:
        logger.error("❌ Comprehensive Utility Integration Testing FAILED!")
        logger.error(f"❌ Error: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()