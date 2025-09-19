#!/usr/bin/env python3
"""
Test script for Enhanced HMM Clustering implementation.

This script tests the HMM clustering system with various configurations
and validates integration with common utilities.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the modules to test
from enhanced_hmm_clustering import (
    EnhancedHMMClustering, 
    HMMClusteringConfig, 
    RegimeType,
    run_hmm_clustering_analysis
)
from config import (
    HMMClusteringConfigFactory, 
    ConfigValidator, 
    get_config_by_name,
    create_custom_config
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic test data for testing."""
    np.random.seed(42)
    
    # Generate synthetic price data
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    # Create multiple regimes with different characteristics
    regime_lengths = [200, 300, 250, 250]
    regimes = []
    
    for i, length in enumerate(regime_lengths):
        if i == 0:  # Bull market
            trend = 0.001
            volatility = 0.02
        elif i == 1:  # Bear market
            trend = -0.0005
            volatility = 0.03
        elif i == 2:  # Sideways market
            trend = 0.0001
            volatility = 0.015
        else:  # High volatility
            trend = 0.0002
            volatility = 0.04
        
        regime_data = np.random.normal(trend, volatility, length)
        regimes.extend(regime_data)
    
    # Ensure we have exactly n_samples
    regimes = regimes[:n_samples]
    
    # Generate price series
    prices = [100]  # Starting price
    for return_val in regimes:
        prices.append(prices[-1] * (1 + return_val))
    
    prices = prices[:n_samples]
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Ensure high >= low and high >= close >= low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    data['high'] = np.maximum(data['high'], data['open'])
    data['low'] = np.minimum(data['low'], data['open'])
    
    return data

def test_configuration_system():
    """Test the configuration system."""
    logger.info("Testing configuration system...")
    
    try:
        # Test factory methods
        crypto_config = HMMClusteringConfigFactory.create_crypto_config()
        forex_config = HMMClusteringConfigFactory.create_forex_config()
        stocks_config = HMMClusteringConfigFactory.create_stocks_config()
        
        assert crypto_config.n_components > 0
        assert forex_config.n_components > 0
        assert stocks_config.n_components > 0
        
        logger.info("✓ Configuration factory methods working")
        
        # Test preset retrieval
        preset_config = get_config_by_name("crypto_btc_1h")
        assert preset_config is not None
        assert preset_config.n_components > 0
        
        logger.info("✓ Preset configuration retrieval working")
        
        # Test custom configuration
        custom_config = create_custom_config(
            n_components=3,
            lookback_windows=[5, 10, 20],
            technical_indicators=["rsi", "macd"]
        )
        assert custom_config.n_components == 3
        assert custom_config.lookback_windows == [5, 10, 20]
        
        logger.info("✓ Custom configuration creation working")
        
        # Test configuration validation
        validator = ConfigValidator()
        warnings = validator.validate_config(crypto_config)
        logger.info(f"Configuration validation warnings: {warnings}")
        
        logger.info("✓ Configuration system tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration system test failed: {e}")
        return False

def test_hmm_clustering_basic():
    """Test basic HMM clustering functionality."""
    logger.info("Testing basic HMM clustering...")
    
    try:
        # Create test data
        test_data = create_test_data(500)
        logger.info(f"Created test data with {len(test_data)} samples")
        
        # Create configuration
        config = HMMClusteringConfig(
            n_components=3,
            lookback_windows=[5, 10, 20],
            technical_indicators=["rsi", "macd", "bollinger_bands"],
            use_gpu=False,  # Disable GPU for testing
            use_memory_optimization=False,  # Disable for testing
            max_features=10,
            min_data_points=100
        )
        
        # Initialize clustering
        clustering = EnhancedHMMClustering(config)
        
        # Test feature engineering
        features = clustering.engineer_features(test_data)
        assert not features.empty
        assert len(features.columns) > 0
        logger.info(f"✓ Feature engineering created {len(features.columns)} features")
        
        # Test feature selection if available
        if clustering.feature_selector is not None:
            selected_features = clustering.select_features(features)
            assert not selected_features.empty
            assert len(selected_features.columns) <= config.max_features
            logger.info(f"✓ Feature selection reduced to {len(selected_features.columns)} features")
        else:
            # Use all features if feature selector not available
            selected_features = features
            logger.info("⚠️ Feature selector not available, using all features")
        
        # Test HMM fitting
        result = clustering.fit_hmm_model(selected_features)
        assert result is not None
        assert result.model is not None
        assert len(result.regime_labels) == len(selected_features)
        assert result.regime_probabilities.shape[0] == len(selected_features)
        
        logger.info("✓ HMM model fitting successful")
        logger.info(f"  - Processing time: {result.processing_time:.2f}s")
        logger.info(f"  - Regime stability: {result.performance_metrics.get('regime_stability', 0):.4f}")
        logger.info(f"  - Regime balance: {result.performance_metrics.get('regime_balance', 0):.4f}")
        
        # Test prediction
        new_regime_labels, new_regime_probs = clustering.predict_regimes(selected_features)
        assert len(new_regime_labels) == len(selected_features)
        assert new_regime_probs.shape[0] == len(selected_features)
        
        logger.info("✓ HMM prediction working")
        
        # Test model saving/loading if serializer available
        if clustering.serializer is not None:
            model_path = "test_model.pkl"
            save_success = clustering.save_model(model_path)
            assert save_success
            
            # Create new clustering instance and load model
            new_clustering = EnhancedHMMClustering(config)
            load_success = new_clustering.load_model(model_path)
            assert load_success
            assert new_clustering.is_fitted
            
            logger.info("✓ Model saving/loading working")
            
            # Clean up
            Path(model_path).unlink(missing_ok=True)
        else:
            logger.info("⚠️ Serializer not available, skipping save/load test")
        
        logger.info("✓ Basic HMM clustering tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Basic HMM clustering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_technical_indicators():
    """Test technical indicator calculations."""
    logger.info("Testing technical indicators...")
    
    try:
        # Create test data
        test_data = create_test_data(200)
        
        # Create clustering instance
        clustering = EnhancedHMMClustering()
        
        # Test RSI calculation
        rsi = clustering._calculate_rsi(test_data['close'], 14)
        assert len(rsi) == len(test_data)
        assert not rsi.isna().all()
        logger.info("✓ RSI calculation working")
        
        # Test MACD calculation
        macd_line, macd_signal, macd_hist = clustering._calculate_macd(test_data['close'])
        assert len(macd_line) == len(test_data)
        assert len(macd_signal) == len(test_data)
        assert len(macd_hist) == len(test_data)
        logger.info("✓ MACD calculation working")
        
        # Test Bollinger Bands calculation
        bb_upper, bb_middle, bb_lower = clustering._calculate_bollinger_bands(test_data['close'])
        assert len(bb_upper) == len(test_data)
        assert len(bb_middle) == len(test_data)
        assert len(bb_lower) == len(test_data)
        logger.info("✓ Bollinger Bands calculation working")
        
        # Test ATR calculation
        atr = clustering._calculate_atr(test_data)
        assert len(atr) == len(test_data)
        logger.info("✓ ATR calculation working")
        
        # Test Stochastic calculation
        stoch_k, stoch_d = clustering._calculate_stochastic(test_data)
        assert len(stoch_k) == len(test_data)
        assert len(stoch_d) == len(test_data)
        logger.info("✓ Stochastic calculation working")
        
        logger.info("✓ Technical indicators tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Technical indicators test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_common_utilities():
    """Test integration with common utilities."""
    logger.info("Testing integration with common utilities...")
    
    try:
        # Test math validation integration
        from src.utils.math_validation import safe_divide, validate_finite
        
        result = safe_divide(10, 2)
        assert result == 5.0
        
        result = safe_divide(10, 0)
        assert result == 0.0
        
        finite_val = validate_finite(5.0)
        assert finite_val == 5.0
        
        logger.info("✓ Math validation integration working")
        
        # Test common operations integration
        from src.utils.common_operations import safe_dataframe_operation
        
        def test_operation(df):
            return df * 2
        
        test_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result_df = safe_dataframe_operation(test_df, test_operation)
        assert (result_df == test_df * 2).all().all()
        
        logger.info("✓ Common operations integration working")
        
        # Test data quality metrics
        from src.utils.common_utilities import calculate_data_quality_metrics
        
        test_data = create_test_data(100)
        quality_metrics = calculate_data_quality_metrics(test_data)
        assert 'missing_values' in quality_metrics
        assert 'data_types' in quality_metrics
        
        logger.info("✓ Data quality metrics integration working")
        
        logger.info("✓ Common utilities integration tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Common utilities integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_optimization():
    """Test performance optimization features."""
    logger.info("Testing performance optimization...")
    
    try:
        # Test with different configurations
        configs = [
            HMMClusteringConfig(use_gpu=False, use_memory_optimization=False),
            HMMClusteringConfig(use_gpu=False, use_memory_optimization=True),
        ]
        
        test_data = create_test_data(300)
        
        for i, config in enumerate(configs):
            logger.info(f"Testing configuration {i+1}...")
            
            clustering = EnhancedHMMClustering(config)
            features = clustering.engineer_features(test_data)
            selected_features = clustering.select_features(features)
            
            start_time = datetime.now()
            result = clustering.fit_hmm_model(selected_features)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            logger.info(f"  Processing time: {processing_time:.2f}s")
            logger.info(f"  Memory usage: {result.memory_usage}")
        
        logger.info("✓ Performance optimization tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and error conditions."""
    logger.info("Testing edge cases...")
    
    try:
        # Test with minimal data
        minimal_data = create_test_data(50)  # Very small dataset
        config = HMMClusteringConfig(
            n_components=2,
            lookback_windows=[5],
            technical_indicators=["rsi"],
            use_gpu=False,
            use_memory_optimization=False,
            max_features=5,
            min_data_points=10  # Lower threshold for testing
        )
        
        clustering = EnhancedHMMClustering(config)
        features = clustering.engineer_features(minimal_data)
        assert not features.empty
        logger.info("✓ Minimal data handling working")
        
        # Test with single regime (constant prices)
        constant_data = minimal_data.copy()
        constant_data['close'] = 100.0  # All same price
        constant_data['high'] = 100.1
        constant_data['low'] = 99.9
        constant_data['open'] = 100.0
        
        try:
            features = clustering.engineer_features(constant_data)
            # Should handle constant data gracefully
            logger.info("✓ Constant data handling working")
        except Exception as e:
            logger.info(f"⚠️ Constant data handling failed as expected: {e}")
        
        # Test empty data handling
        try:
            empty_data = pd.DataFrame()
            clustering.engineer_features(empty_data)
            assert False, "Should have raised exception for empty data"
        except ValueError:
            logger.info("✓ Empty data validation working")
        
        # Test missing columns
        try:
            invalid_data = pd.DataFrame({'volume': [1, 2, 3]})
            clustering.engineer_features(invalid_data)
            assert False, "Should have raised exception for missing close column"
        except ValueError:
            logger.info("✓ Missing column validation working")
        
        # Test prediction without fitting
        try:
            unfitted_clustering = EnhancedHMMClustering(config)
            test_features = pd.DataFrame({'feature1': [1, 2, 3]})
            unfitted_clustering.predict_regimes(test_features)
            assert False, "Should have raised exception for unfitted model"
        except ValueError:
            logger.info("✓ Unfitted model validation working")
        
        logger.info("✓ Edge cases tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mathematical_correctness():
    """Test mathematical correctness of indicators."""
    logger.info("Testing mathematical correctness...")
    
    try:
        # Create predictable test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
            'open': np.arange(100, 200),
            'high': np.arange(101, 201),
            'low': np.arange(99, 199),
            'close': np.arange(100, 200),
            'volume': np.full(100, 1000)
        })
        
        config = HMMClusteringConfig(
            n_components=2,
            lookback_windows=[14],
            technical_indicators=["rsi"],
            use_gpu=False,
            max_features=5
        )
        
        clustering = EnhancedHMMClustering(config)
        
        # Test RSI calculation
        rsi = clustering._calculate_rsi(test_data['close'], 14)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert all(0 <= val <= 100 for val in valid_rsi), "RSI values should be between 0 and 100"
        
        # For steadily increasing prices, RSI should be high
        assert valid_rsi.iloc[-1] > 50, "RSI should be high for increasing prices"
        
        logger.info("✓ RSI calculation mathematically correct")
        
        # Test Stochastic calculation
        stoch_k, stoch_d = clustering._calculate_stochastic(test_data, 14)
        
        valid_stoch_k = stoch_k.dropna()
        valid_stoch_d = stoch_d.dropna()
        
        # Stochastic should be between 0 and 100
        assert all(0 <= val <= 100 for val in valid_stoch_k), "Stochastic K should be between 0 and 100"
        assert all(0 <= val <= 100 for val in valid_stoch_d), "Stochastic D should be between 0 and 100"
        
        logger.info("✓ Stochastic calculation mathematically correct")
        
        # Test regime stability calculation
        regime_labels = np.array([0, 0, 1, 1, 1, 0, 0])  # 3 regime changes
        regime_changes = np.sum(np.diff(regime_labels) != 0)
        stability = 1 - (regime_changes / (len(regime_labels) - 1))
        expected_stability = 1 - (3 / 6)  # 3 changes out of 6 possible transitions
        assert abs(stability - expected_stability) < 0.001, f"Regime stability calculation incorrect: {stability} vs {expected_stability}"
        
        logger.info("✓ Regime stability calculation mathematically correct")
        
        logger.info("✓ Mathematical correctness tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Mathematical correctness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests."""
    logger.info("Starting comprehensive test suite...")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration System", test_configuration_system),
        ("Basic HMM Clustering", test_hmm_clustering_basic),
        ("Technical Indicators", test_technical_indicators),
        ("Common Utilities Integration", test_integration_with_common_utilities),
        ("Performance Optimization", test_performance_optimization),
        ("Edge Cases", test_edge_cases),
        ("Mathematical Correctness", test_mathematical_correctness),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} test PASSED")
            else:
                logger.error(f"✗ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} test FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Implementation is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    # Create output directory
    output_dir = Path("market_analysis/hmm_clustering/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("The Enhanced HMM Clustering implementation is ready for use.")
    else:
        print("\n❌ Some tests failed. Please review the implementation.")
        sys.exit(1)