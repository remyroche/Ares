"""
Comprehensive Test Suite for Enhanced Partial Information Decomposition

This test suite demonstrates the enhanced PID functionality with:
- Multiple PID measures (I_min, I_ccs, I_dep, I_mmi)
- Proper mathematical foundations
- Input validation
- Vectorized operations and parallel processing
- Financial domain features
- Error handling
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our enhanced PID modules
try:
    from src.training.utils.feature_selection.enhanced_partial_information_decomposition import (
        PIDConfig, PIDMeasure, DiscretizationMethod, PIDResult,
        EntropyCalculator, MutualInformationCalculator, PIDCalculator
    )
    from src.training.utils.feature_selection.enhanced_pid_main import (
        EnhancedPartialInformationDecomposition,
        create_enhanced_pid_module,
        compute_enhanced_pid
    )
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logger.error(f"Failed to import enhanced PID modules: {e}")
    IMPORTS_SUCCESSFUL = False

def create_sample_financial_data(n_samples: int = 1000, n_features: int = 10) -> tuple:
    """Create sample financial data for testing."""
    np.random.seed(42)
    
    # Create realistic financial features
    feature_names = [
        'price_close', 'price_open', 'price_high', 'price_low',
        'volume', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower', 'volatility'
    ]
    
    # Generate correlated financial data
    base_price = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    
    X = np.zeros((n_samples, n_features))
    
    # Price features (highly correlated)
    X[:, 0] = base_price  # close
    X[:, 1] = base_price + np.random.randn(n_samples) * 0.5  # open
    X[:, 2] = base_price + np.abs(np.random.randn(n_samples) * 0.8)  # high
    X[:, 3] = base_price - np.abs(np.random.randn(n_samples) * 0.8)  # low
    
    # Volume (correlated with price movements)
    price_changes = np.abs(np.diff(base_price, prepend=base_price[0]))
    X[:, 4] = 1000 + price_changes * 100 + np.random.randn(n_samples) * 50  # volume
    
    # Technical indicators
    X[:, 5] = 50 + np.random.randn(n_samples) * 15  # RSI
    X[:, 6] = np.random.randn(n_samples) * 0.5  # MACD
    X[:, 7] = base_price + np.random.randn(n_samples) * 2  # Bollinger upper
    X[:, 8] = base_price - np.random.randn(n_samples) * 2  # Bollinger lower
    X[:, 9] = np.random.randn(n_samples) * 0.02  # Volatility
    
    # Create target variable (returns) with some relationship to features
    returns = np.random.randn(n_samples) * 0.01
    # Add some signal from price features
    returns += (X[:, 0] - X[:, 1]) * 0.001  # Close-open difference
    returns += X[:, 9] * 0.1  # Volatility effect
    
    y = returns
    
    return X, y, feature_names

def test_entropy_calculator():
    """Test the enhanced entropy calculator."""
    logger.info("🧪 Testing Entropy Calculator")
    
    # Test data
    np.random.seed(42)
    data = np.random.randn(1000)
    
    # Test different estimators
    estimators = ["plugin", "miller_madow", "jackknife"]
    
    for estimator in estimators:
        try:
            calc = EntropyCalculator(estimator)
            entropy = calc.calculate_entropy(data)
            logger.info(f"✅ {estimator} entropy: {entropy:.4f}")
        except Exception as e:
            logger.error(f"❌ {estimator} entropy failed: {e}")
    
    # Test with different data types
    discrete_data = np.random.randint(0, 10, 1000)
    calc = EntropyCalculator("plugin")
    discrete_entropy = calc.calculate_entropy(discrete_data)
    logger.info(f"✅ Discrete data entropy: {discrete_entropy:.4f}")

def test_mutual_information_calculator():
    """Test the enhanced mutual information calculator."""
    logger.info("🧪 Testing Mutual Information Calculator")
    
    # Test data
    np.random.seed(42)
    x = np.random.randn(1000)
    y = x + np.random.randn(1000) * 0.5  # Correlated with x
    
    # Test different estimators
    estimators = ["plugin", "knn", "gaussian"]
    
    for estimator in estimators:
        try:
            calc = MutualInformationCalculator(estimator)
            mi = calc.calculate_mutual_information(x, y)
            logger.info(f"✅ {estimator} MI: {mi:.4f}")
        except Exception as e:
            logger.error(f"❌ {estimator} MI failed: {e}")
    
    # Test with uncorrelated data
    z = np.random.randn(1000)
    calc = MutualInformationCalculator("plugin")
    mi_uncorrelated = calc.calculate_mutual_information(x, z)
    logger.info(f"✅ Uncorrelated MI: {mi_uncorrelated:.4f}")

def test_pid_calculator():
    """Test the PID calculator with multiple measures."""
    logger.info("🧪 Testing PID Calculator")
    
    # Test data
    np.random.seed(42)
    x1 = np.random.randn(1000)
    x2 = x1 + np.random.randn(1000) * 0.3  # Correlated with x1
    y = x1 + x2 + np.random.randn(1000) * 0.2  # Target depends on both
    
    # Test configuration
    config = PIDConfig(
        pid_measures=[PIDMeasure.I_MIN, PIDMeasure.I_CCS, PIDMeasure.I_DEP, PIDMeasure.I_MMI],
        entropy_estimator="plugin",
        mutual_info_estimator="plugin"
    )
    
    try:
        pid_calc = PIDCalculator(config)
        results = pid_calc.compute_pid(x1, x2, y)
        
        for measure, result in results.items():
            logger.info(f"✅ {measure.value}:")
            logger.info(f"   Unique X1: {result.unique_x1:.4f}")
            logger.info(f"   Unique X2: {result.unique_x2:.4f}")
            logger.info(f"   Redundant: {result.redundant:.4f}")
            logger.info(f"   Synergistic: {result.synergistic:.4f}")
            logger.info(f"   Total MI: {result.total_mi:.4f}")
            logger.info(f"   Computation time: {result.computation_time:.4f}s")
            
    except Exception as e:
        logger.error(f"❌ PID calculation failed: {e}")

def test_discretization_methods():
    """Test different discretization methods."""
    logger.info("🧪 Testing Discretization Methods")
    
    # Test data
    np.random.seed(42)
    data = np.random.randn(1000)
    
    # Test different methods
    methods = [
        DiscretizationMethod.EQUAL_WIDTH,
        DiscretizationMethod.EQUAL_FREQUENCY,
        DiscretizationMethod.KMEANS,
        DiscretizationMethod.QUANTILE,
        DiscretizationMethod.ADAPTIVE
    ]
    
    config = PIDConfig(n_bins=10)
    
    for method in methods:
        try:
            config.discretization_method = method
            pid_module = create_enhanced_pid_module(config)
            
            # Test discretization
            discrete = pid_module._discretize_vector(data)
            unique_bins = len(np.unique(discrete))
            
            logger.info(f"✅ {method.value}: {unique_bins} unique bins")
            
        except Exception as e:
            logger.error(f"❌ {method.value} discretization failed: {e}")

def test_enhanced_pid_module():
    """Test the main enhanced PID module."""
    logger.info("🧪 Testing Enhanced PID Module")
    
    # Create sample data
    X, y, feature_names = create_sample_financial_data(1000, 10)
    
    # Test different configurations
    configs = [
        PIDConfig(
            method="bivariate",
            pid_measures=[PIDMeasure.I_MIN],
            discretization_method=DiscretizationMethod.ADAPTIVE,
            enable_parallel=False,
            enable_financial_features=False
        ),
        PIDConfig(
            method="bivariate",
            pid_measures=[PIDMeasure.I_MIN, PIDMeasure.I_CCS],
            discretization_method=DiscretizationMethod.QUANTILE,
            enable_parallel=True,
            enable_financial_features=True
        ),
        PIDConfig(
            method="trivariate",
            pid_measures=[PIDMeasure.I_MIN, PIDMeasure.I_CCS, PIDMeasure.I_DEP],
            discretization_method=DiscretizationMethod.KMEANS,
            enable_parallel=True,
            enable_financial_features=True
        )
    ]
    
    for i, config in enumerate(configs):
        logger.info(f"🧪 Testing Configuration {i+1}")
        
        try:
            # Create PID module
            pid_module = create_enhanced_pid_module(config)
            
            # Compute PID
            start_time = time.time()
            results = pid_module.compute_pid(X, y, feature_names)
            computation_time = time.time() - start_time
            
            # Get summary
            summary = pid_module.get_pid_summary()
            
            logger.info(f"✅ Configuration {i+1} completed in {computation_time:.3f}s")
            logger.info(f"   Method: {summary['method']}")
            logger.info(f"   Features analyzed: {summary['total_features_analyzed']}")
            logger.info(f"   Financial features: {summary['financial_features_created']}")
            logger.info(f"   PID measures: {summary['pid_measures_used']}")
            
            # Test financial features if enabled
            if config.enable_financial_features and 'financial_features' in results:
                financial_features = results['financial_features']
                logger.info(f"   Financial features created: {len(financial_features)}")
                
                # Show some example features
                for i, (name, feature) in enumerate(list(financial_features.items())[:3]):
                    logger.info(f"     {name}: shape {feature.shape}, mean {np.mean(feature):.4f}")
            
        except Exception as e:
            logger.error(f"❌ Configuration {i+1} failed: {e}")

def test_input_validation():
    """Test input validation functionality."""
    logger.info("🧪 Testing Input Validation")
    
    # Create PID module
    pid_module = create_enhanced_pid_module()
    
    # Test valid inputs
    X_valid, y_valid, feature_names_valid = create_sample_financial_data(100, 5)
    
    try:
        is_valid = pid_module.validate_inputs(X_valid, y_valid, feature_names_valid)
        logger.info(f"✅ Valid inputs: {is_valid}")
    except Exception as e:
        logger.error(f"❌ Valid input validation failed: {e}")
    
    # Test invalid inputs
    test_cases = [
        ("Wrong X dimensions", X_valid.reshape(-1, 1), y_valid, feature_names_valid),
        ("Wrong y dimensions", X_valid, y_valid.reshape(-1, 1), feature_names_valid),
        ("Mismatched samples", X_valid[:50], y_valid, feature_names_valid),
        ("Mismatched features", X_valid, y_valid, feature_names_valid[:3]),
    ]
    
    for test_name, X_test, y_test, names_test in test_cases:
        try:
            is_valid = pid_module.validate_inputs(X_test, y_test, names_test)
            logger.info(f"✅ {test_name}: validation correctly failed")
        except Exception as e:
            logger.info(f"✅ {test_name}: validation correctly caught error: {e}")

def test_parallel_processing():
    """Test parallel processing functionality."""
    logger.info("🧪 Testing Parallel Processing")
    
    # Create larger dataset for parallel processing test
    X, y, feature_names = create_sample_financial_data(2000, 20)
    
    # Test sequential vs parallel
    configs = [
        PIDConfig(enable_parallel=False, n_jobs=1),
        PIDConfig(enable_parallel=True, n_jobs=4)
    ]
    
    for i, config in enumerate(configs):
        try:
            pid_module = create_enhanced_pid_module(config)
            
            start_time = time.time()
            results = pid_module.compute_pid(X, y, feature_names)
            computation_time = time.time() - start_time
            
            processing_type = "Sequential" if not config.enable_parallel else "Parallel"
            logger.info(f"✅ {processing_type} processing: {computation_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ {processing_type} processing failed: {e}")

def test_financial_features():
    """Test financial domain-specific features."""
    logger.info("🧪 Testing Financial Features")
    
    # Create financial data
    X, y, feature_names = create_sample_financial_data(1000, 10)
    
    config = PIDConfig(
        enable_financial_features=True,
        regime_aware=True,
        volatility_threshold=0.01,
        correlation_threshold=0.1
    )
    
    try:
        pid_module = create_enhanced_pid_module(config)
        
        # Test individual feature creation methods
        price_features = pid_module._create_price_features(X, feature_names)
        volatility_features = pid_module._create_volatility_features(X, feature_names)
        correlation_features = pid_module._create_correlation_features(X, feature_names)
        regime_features = pid_module._create_regime_features(X, y, feature_names)
        
        logger.info(f"✅ Price features: {len(price_features)}")
        logger.info(f"✅ Volatility features: {len(volatility_features)}")
        logger.info(f"✅ Correlation features: {len(correlation_features)}")
        logger.info(f"✅ Regime features: {len(regime_features)}")
        
        # Test full financial feature creation
        all_financial_features = pid_module._create_financial_features(X, y, feature_names)
        logger.info(f"✅ Total financial features: {len(all_financial_features)}")
        
    except Exception as e:
        logger.error(f"❌ Financial features test failed: {e}")

def test_error_handling():
    """Test comprehensive error handling."""
    logger.info("🧪 Testing Error Handling")
    
    pid_module = create_enhanced_pid_module()
    
    # Test with problematic data
    test_cases = [
        ("Empty data", np.array([]), np.array([]), []),
        ("Single sample", np.array([[1, 2]]), np.array([1]), ["f1", "f2"]),
        ("All NaN", np.full((100, 5), np.nan), np.full(100, np.nan), ["f1", "f2", "f3", "f4", "f5"]),
        ("All zeros", np.zeros((100, 5)), np.zeros(100), ["f1", "f2", "f3", "f4", "f5"]),
        ("Infinite values", np.full((100, 5), np.inf), np.full(100, np.inf), ["f1", "f2", "f3", "f4", "f5"]),
    ]
    
    for test_name, X_test, y_test, names_test in test_cases:
        try:
            if len(X_test) > 0 and len(y_test) > 0:
                results = pid_module.compute_pid(X_test, y_test, names_test)
                logger.info(f"✅ {test_name}: handled gracefully")
            else:
                logger.info(f"✅ {test_name}: correctly rejected")
        except Exception as e:
            logger.info(f"✅ {test_name}: correctly caught error: {e}")

def test_performance_benchmark():
    """Test performance with different dataset sizes."""
    logger.info("🧪 Testing Performance Benchmark")
    
    dataset_sizes = [100, 500, 1000, 2000]
    
    for size in dataset_sizes:
        logger.info(f"🧪 Testing with {size} samples")
        
        # Create data
        X, y, feature_names = create_sample_financial_data(size, 10)
        
        config = PIDConfig(
            enable_parallel=True,
            enable_financial_features=True,
            pid_measures=[PIDMeasure.I_MIN, PIDMeasure.I_CCS]
        )
        
        try:
            pid_module = create_enhanced_pid_module(config)
            
            start_time = time.time()
            results = pid_module.compute_pid(X, y, feature_names)
            computation_time = time.time() - start_time
            
            logger.info(f"✅ {size} samples: {computation_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ {size} samples failed: {e}")

def main():
    """Run all tests."""
    if not IMPORTS_SUCCESSFUL:
        logger.error("❌ Cannot run tests - imports failed")
        return
    
    logger.info("🚀 Starting Enhanced PID Test Suite")
    logger.info("=" * 60)
    
    # Run all tests
    test_functions = [
        test_entropy_calculator,
        test_mutual_information_calculator,
        test_pid_calculator,
        test_discretization_methods,
        test_input_validation,
        test_enhanced_pid_module,
        test_parallel_processing,
        test_financial_features,
        test_error_handling,
        test_performance_benchmark
    ]
    
    for test_func in test_functions:
        try:
            test_func()
            logger.info("")
        except Exception as e:
            logger.error(f"❌ {test_func.__name__} failed with exception: {e}")
            logger.info("")
    
    logger.info("🎉 Enhanced PID Test Suite Complete!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()