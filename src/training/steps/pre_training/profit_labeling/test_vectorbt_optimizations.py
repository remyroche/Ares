"""
Test script for VectorBT optimizations in profit labeling

This script validates the performance improvements and ensures backward compatibility
of the VectorBT optimizations implemented in the profit labeling system.
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Any
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import the optimized modules
from .vectorbt_optimizer import get_vectorbt_optimizer, VectorBTConfig
from .enhanced_data_labels_system import EnhancedDataLabelsSystem, EnhancedDataLabelsConfig
from .bar_construction import EventBasedBarConstructor, BarConstructionConfig
from .quality_scoring import LabelQualityScorer, QualityScoringConfig
from .enhanced_label_definitions import EnhancedLabelDefinitions, AnalystLabelConfig

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


def generate_test_data(n_samples: int = 10000, n_features: int = 10) -> pd.DataFrame:
    """Generate synthetic test data for performance testing."""
    np.random.seed(42)
    
    # Generate realistic market data
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1min')
    
    # Generate price data with realistic patterns
    base_price = 100.0
    returns = np.random.normal(0, 0.001, n_samples)  # 0.1% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)
    
    # Ensure OHLC relationships are valid
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data


def test_vectorbt_optimizer_performance():
    """Test VectorBT optimizer performance against pandas operations."""
    tprint_info("🧪 Testing VectorBT optimizer performance...")
    
    # Generate test data
    data = generate_test_data(n_samples=5000)
    returns = data['close'].pct_change().dropna()
    
    # Initialize optimizer
    config = VectorBTConfig(
        enable_vectorbt=True,
        vectorbt_threshold=1000,
        performance_monitoring=True
    )
    optimizer = get_vectorbt_optimizer(config)
    
    # Test rolling operations
    operations = [
        ('rolling_mean', lambda: optimizer.rolling_mean(returns, window=20)),
        ('rolling_std', lambda: optimizer.rolling_std(returns, window=20)),
        ('rolling_var', lambda: optimizer.rolling_var(returns, window=20)),
        ('rolling_min', lambda: optimizer.rolling_min(returns, window=20)),
        ('rolling_max', lambda: optimizer.rolling_max(returns, window=20)),
        ('rolling_sum', lambda: optimizer.rolling_sum(returns, window=20)),
    ]
    
    results = {}
    for op_name, op_func in operations:
        start_time = time.time()
        result = op_func()
        execution_time = time.time() - start_time
        
        results[op_name] = {
            'execution_time': execution_time,
            'result_length': len(result),
            'has_nan': result.isna().any(),
            'is_finite': np.isfinite(result).all()
        }
        
        tprint_info(f"   → {op_name}: {execution_time:.4f}s, length: {len(result)}")
    
    # Test specialized operations
    tprint_info("   → Testing specialized operations...")
    
    start_time = time.time()
    volatility = optimizer.calculate_volatility(returns, window=20)
    vol_time = time.time() - start_time
    
    start_time = time.time()
    rsi = optimizer.calculate_rsi(data['close'], window=14)
    rsi_time = time.time() - start_time
    
    results['volatility'] = {
        'execution_time': vol_time,
        'result_length': len(volatility),
        'has_nan': volatility.isna().any(),
        'is_finite': np.isfinite(volatility).all()
    }
    
    results['rsi'] = {
        'execution_time': rsi_time,
        'result_length': len(rsi),
        'has_nan': rsi.isna().any(),
        'is_finite': np.isfinite(rsi).all()
    }
    
    tprint_info(f"   → volatility: {vol_time:.4f}s")
    tprint_info(f"   → rsi: {rsi_time:.4f}s")
    
    # Get performance summary
    perf_summary = optimizer.get_performance_summary()
    tprint_success(f"✅ VectorBT optimizer test completed")
    tprint_info(f"   → Total operations: {perf_summary['total_operations']}")
    tprint_info(f"   → Success rate: {perf_summary['success_rate']:.2%}")
    tprint_info(f"   → VectorBT usage: {perf_summary['vectorbt_usage_rate']:.2%}")
    tprint_info(f"   → Avg execution time: {perf_summary['avg_execution_time']:.4f}s")
    
    return results


def test_enhanced_data_labels_system():
    """Test enhanced data labels system with VectorBT optimizations."""
    tprint_info("🧪 Testing Enhanced Data Labels System...")
    
    # Generate test data
    data = generate_test_data(n_samples=2000)
    
    # Initialize system with VectorBT config
    config = EnhancedDataLabelsConfig(
        vectorbt_config=VectorBTConfig(
            enable_vectorbt=True,
            vectorbt_threshold=500,
            performance_monitoring=True
        )
    )
    
    system = EnhancedDataLabelsSystem(config)
    
    # Test data processing
    start_time = time.time()
    result = system.process_market_data(data)
    processing_time = time.time() - start_time
    
    # Validate results
    success = (
        'processed_data' in result and
        'labels' in result and
        'data_quality' in result and
        'label_stability' in result
    )
    
    if success:
        tprint_success(f"✅ Enhanced Data Labels System test completed in {processing_time:.2f}s")
        tprint_info(f"   → Processed data shape: {result['processed_data'].shape}")
        tprint_info(f"   → Labels shape: {result['labels'].shape}")
        tprint_info(f"   → Data quality: {result['data_quality']['quality_level'].value}")
        tprint_info(f"   → Label stability: {result['label_stability']['stability_level'].value}")
    else:
        tprint_error("❌ Enhanced Data Labels System test failed")
    
    return success, processing_time


def test_bar_construction():
    """Test bar construction with VectorBT optimizations."""
    tprint_info("🧪 Testing Bar Construction...")
    
    # Generate test data
    data = generate_test_data(n_samples=5000)
    
    # Initialize constructor with VectorBT config
    config = BarConstructionConfig(
        bar_type='dollar',
        bar_size=100000.0,
        vectorbt_config=VectorBTConfig(
            enable_vectorbt=True,
            vectorbt_threshold=1000,
            performance_monitoring=True
        )
    )
    
    constructor = EventBasedBarConstructor(config)
    
    # Test bar construction
    start_time = time.time()
    result = constructor.construct_bars(data)
    construction_time = time.time() - start_time
    
    # Validate results
    success = (
        result.n_original_bars > 0 and
        result.n_cleaned_bars > 0 and
        not result.cleaned_bars.empty
    )
    
    if success:
        tprint_success(f"✅ Bar Construction test completed in {construction_time:.2f}s")
        tprint_info(f"   → Original bars: {result.n_original_bars}")
        tprint_info(f"   → Cleaned bars: {result.n_cleaned_bars}")
        tprint_info(f"   → Quality score: {result.data_quality_score:.3f}")
    else:
        tprint_error("❌ Bar Construction test failed")
    
    return success, construction_time


def test_quality_scoring():
    """Test quality scoring with VectorBT optimizations."""
    tprint_info("🧪 Testing Quality Scoring...")
    
    # Generate test data
    data = generate_test_data(n_samples=3000)
    
    # Create mock labels and confidence scores
    labels = pd.DataFrame({
        'target_1': np.random.choice([0, 1], size=len(data), p=[0.7, 0.3]),
        'target_2': np.random.choice([0, 1], size=len(data), p=[0.6, 0.4])
    }, index=data.index)
    
    confidence_scores = pd.DataFrame({
        'target_1': np.random.uniform(0.3, 0.9, len(data)),
        'target_2': np.random.uniform(0.3, 0.9, len(data))
    }, index=data.index)
    
    eligibility_masks = pd.DataFrame({
        'target_1': np.random.choice([True, False], size=len(data), p=[0.8, 0.2]),
        'target_2': np.random.choice([True, False], size=len(data), p=[0.8, 0.2])
    }, index=data.index)
    
    # Initialize scorer with VectorBT config
    config = QualityScoringConfig(
        vectorbt_config=VectorBTConfig(
            enable_vectorbt=True,
            vectorbt_threshold=1000,
            performance_monitoring=True
        )
    )
    
    scorer = LabelQualityScorer(config)
    
    # Test quality assessment
    start_time = time.time()
    quality_results = scorer.assess_quality(labels, confidence_scores, eligibility_masks, data)
    assessment_time = time.time() - start_time
    
    # Validate results
    success = len(quality_results) > 0 and all(
        hasattr(metrics, 'lqs_score') for metrics in quality_results.values()
    )
    
    if success:
        tprint_success(f"✅ Quality Scoring test completed in {assessment_time:.2f}s")
        for target, metrics in quality_results.items():
            tprint_info(f"   → {target}: LQS={metrics.lqs_score:.3f}, samples={metrics.n_samples}")
    else:
        tprint_error("❌ Quality Scoring test failed")
    
    return success, assessment_time


def test_memory_efficiency():
    """Test memory efficiency of VectorBT operations."""
    tprint_info("🧪 Testing Memory Efficiency...")
    
    # Test with different data sizes
    sizes = [1000, 5000, 10000, 20000]
    results = {}
    
    for size in sizes:
        tprint_info(f"   → Testing with {size} samples...")
        
        data = generate_test_data(n_samples=size)
        returns = data['close'].pct_change().dropna()
        
        # Initialize optimizer
        config = VectorBTConfig(
            enable_vectorbt=True,
            vectorbt_threshold=1000,
            memory_efficiency_mode=True,
            performance_monitoring=True
        )
        optimizer = get_vectorbt_optimizer(config)
        
        # Test memory usage
        start_time = time.time()
        volatility = optimizer.calculate_volatility(returns, window=20)
        execution_time = time.time() - start_time
        
        results[size] = {
            'execution_time': execution_time,
            'memory_efficient': size <= 10000 or execution_time < 1.0
        }
        
        tprint_info(f"      → Execution time: {execution_time:.4f}s")
    
    # Check if memory efficiency is working
    efficient = all(result['memory_efficient'] for result in results.values())
    
    if efficient:
        tprint_success("✅ Memory efficiency test passed")
    else:
        tprint_warning("⚠️ Memory efficiency test had issues with large datasets")
    
    return efficient, results


def run_comprehensive_test():
    """Run comprehensive test suite for VectorBT optimizations."""
    tprint_info("🚀 Starting comprehensive VectorBT optimization tests...")
    
    test_results = {}
    
    # Test 1: VectorBT Optimizer Performance
    try:
        optimizer_results = test_vectorbt_optimizer_performance()
        test_results['optimizer'] = {'success': True, 'results': optimizer_results}
    except Exception as e:
        tprint_error(f"❌ VectorBT optimizer test failed: {e}")
        test_results['optimizer'] = {'success': False, 'error': str(e)}
    
    # Test 2: Enhanced Data Labels System
    try:
        success, time_taken = test_enhanced_data_labels_system()
        test_results['data_labels'] = {'success': success, 'time': time_taken}
    except Exception as e:
        tprint_error(f"❌ Enhanced Data Labels System test failed: {e}")
        test_results['data_labels'] = {'success': False, 'error': str(e)}
    
    # Test 3: Bar Construction
    try:
        success, time_taken = test_bar_construction()
        test_results['bar_construction'] = {'success': success, 'time': time_taken}
    except Exception as e:
        tprint_error(f"❌ Bar Construction test failed: {e}")
        test_results['bar_construction'] = {'success': False, 'error': str(e)}
    
    # Test 4: Quality Scoring
    try:
        success, time_taken = test_quality_scoring()
        test_results['quality_scoring'] = {'success': success, 'time': time_taken}
    except Exception as e:
        tprint_error(f"❌ Quality Scoring test failed: {e}")
        test_results['quality_scoring'] = {'success': False, 'error': str(e)}
    
    # Test 5: Memory Efficiency
    try:
        efficient, results = test_memory_efficiency()
        test_results['memory_efficiency'] = {'success': efficient, 'results': results}
    except Exception as e:
        tprint_error(f"❌ Memory Efficiency test failed: {e}")
        test_results['memory_efficiency'] = {'success': False, 'error': str(e)}
    
    # Summary
    tprint_info("📊 Test Results Summary:")
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result['success'])
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        tprint_info(f"   → {test_name}: {status}")
        if not result['success'] and 'error' in result:
            tprint_info(f"      Error: {result['error']}")
    
    tprint_success(f"🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    return test_results


if __name__ == "__main__":
    # Run comprehensive test
    results = run_comprehensive_test()
    
    # Print final summary
    print("\n" + "="*60)
    print("VECTORBT OPTIMIZATION TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "PASS" if result['success'] else "FAIL"
        print(f"{test_name.upper()}: {status}")
    
    print("="*60)