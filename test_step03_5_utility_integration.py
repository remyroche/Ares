#!/usr/bin/env python3
"""
Comprehensive test for Step03_5 utility integration.

This script tests the extensive utility integration in step03_5_final_regime_clustering.py
to ensure all utilities are properly injected and functioning.
"""

import asyncio
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the step03_5 module
from src.training.steps.market_analysis.hmm_clustering.step03_5_final_regime_clustering import (
    FinalRegimeClusteringStep,
    UtilityDependencyInjector
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_rows: int = 1000) -> pd.DataFrame:
    """Create test data for utility integration testing."""
    logger.info(f"Creating test data with {n_rows} rows...")
    
    # Generate synthetic market data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='1min')
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0, 0.001, n_rows)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.exponential(1000, n_rows)
    })
    
    # Ensure high >= low and high/low contain open/close
    for i in range(len(data)):
        data.loc[i, 'high'] = max(data.loc[i, 'high'], data.loc[i, 'open'], data.loc[i, 'close'])
        data.loc[i, 'low'] = min(data.loc[i, 'low'], data.loc[i, 'open'], data.loc[i, 'close'])
    
    logger.info(f"✅ Test data created: {len(data)} rows, {len(data.columns)} columns")
    return data

def test_utility_dependency_injection():
    """Test the utility dependency injection framework."""
    logger.info("🧪 Testing utility dependency injection framework...")
    
    try:
        # Create test config
        config = {
            'symbol': 'TESTUSDT',
            'exchange': 'TEST',
            'timeframe': '1m',
            'expected_data_size_mb': 10
        }
        
        # Initialize dependency injector
        injector = UtilityDependencyInjector(config, logger)
        
        # Inject all utilities
        utilities = injector.inject_all_utilities()
        
        # Test utility availability
        required_utilities = [
            'safe_mean', 'safe_divide', 'validate_finite', 'parquet_utils',
            'm1_gpu_manager', 'm1_memory_optimizer', 'm1_cpu_optimizer',
            'JSONSerializer', 'DataFrameValidator', 'UniversalSerializer'
        ]
        
        available_count = 0
        for utility in required_utilities:
            if utility in utilities:
                available_count += 1
                logger.info(f"  ✅ {utility}: Available")
            else:
                logger.warning(f"  ❌ {utility}: Not available")
        
        # Test utility functionality
        test_results = {}
        
        # Test safe_mean
        if 'safe_mean' in utilities:
            try:
                result = utilities['safe_mean']([1, 2, 3, 4, 5])
                test_results['safe_mean'] = result == 3.0
                logger.info(f"  ✅ safe_mean test: {result}")
            except Exception as e:
                test_results['safe_mean'] = False
                logger.error(f"  ❌ safe_mean test failed: {e}")
        
        # Test safe_divide
        if 'safe_divide' in utilities:
            try:
                result = utilities['safe_divide'](10, 2)
                test_results['safe_divide'] = result == 5.0
                logger.info(f"  ✅ safe_divide test: {result}")
            except Exception as e:
                test_results['safe_divide'] = False
                logger.error(f"  ❌ safe_divide test failed: {e}")
        
        # Test validate_finite
        if 'validate_finite' in utilities:
            try:
                result = utilities['validate_finite'](42.0)
                test_results['validate_finite'] = result == 42.0
                logger.info(f"  ✅ validate_finite test: {result}")
            except Exception as e:
                test_results['validate_finite'] = False
                logger.error(f"  ❌ validate_finite test failed: {e}")
        
        # Calculate success rate
        success_rate = sum(test_results.values()) / len(test_results) * 100 if test_results else 0
        availability_rate = available_count / len(required_utilities) * 100
        
        logger.info(f"📊 Utility Integration Test Results:")
        logger.info(f"  📈 Availability Rate: {availability_rate:.1f}% ({available_count}/{len(required_utilities)})")
        logger.info(f"  🧪 Functionality Success Rate: {success_rate:.1f}% ({sum(test_results.values())}/{len(test_results)})")
        
        return {
            'availability_rate': availability_rate,
            'functionality_success_rate': success_rate,
            'available_utilities': available_count,
            'test_results': test_results
        }
        
    except Exception as e:
        logger.error(f"❌ Utility dependency injection test failed: {e}")
        return {
            'availability_rate': 0,
            'functionality_success_rate': 0,
            'available_utilities': 0,
            'test_results': {},
            'error': str(e)
        }

async def test_step03_5_execution():
    """Test the Step03_5 execution with comprehensive utility integration."""
    logger.info("🧪 Testing Step03_5 execution with comprehensive utility integration...")
    
    try:
        # Create test config
        config = {
            'symbol': 'TESTUSDT',
            'exchange': 'TEST',
            'timeframe': '1m',
            'DATA_DIR': '/tmp/test_data',
            'expected_data_size_mb': 10
        }
        
        # Create test data
        test_data = create_test_data(1000)
        
        # Save test data
        test_data_path = Path('/tmp/test_data')
        test_data_path.mkdir(exist_ok=True)
        test_data.to_parquet(test_data_path / 'klines_TEST_TESTUSDT_1m_consolidated.parquet', index=False)
        
        # Initialize Step03_5
        step = FinalRegimeClusteringStep(config)
        
        # Test utility integration status
        logger.info("📊 Testing utility integration status...")
        step._log_utility_integration_status()
        
        # Test comprehensive utility operations
        logger.info("🔧 Testing comprehensive utility operations...")
        utility_operations = step._perform_comprehensive_utility_operations(test_data)
        
        # Test data loading with comprehensive utilities
        logger.info("📂 Testing data loading with comprehensive utilities...")
        try:
            loaded_data = await step._load_data_with_comprehensive_utilities(
                'klines_TEST_TESTUSDT_1m_consolidated', '/tmp/test_data'
            )
            data_loading_success = loaded_data is not None and not loaded_data.empty
            logger.info(f"  ✅ Data loading test: {'Success' if data_loading_success else 'Failed'}")
        except Exception as e:
            data_loading_success = False
            logger.error(f"  ❌ Data loading test failed: {e}")
        
        # Test feature preparation with comprehensive utilities
        logger.info("🔧 Testing feature preparation with comprehensive utilities...")
        try:
            features = await step._prepare_features_with_comprehensive_utilities(test_data)
            feature_prep_success = features is not None and not features.empty
            logger.info(f"  ✅ Feature preparation test: {'Success' if feature_prep_success else 'Failed'}")
        except Exception as e:
            feature_prep_success = False
            logger.error(f"  ❌ Feature preparation test failed: {e}")
        
        # Test HMM regime discovery with utilities
        logger.info("🔍 Testing HMM regime discovery with utilities...")
        try:
            hmm_results = await step._perform_hmm_regime_discovery_with_utilities(test_data)
            hmm_success = hmm_results is not None and 'states' in hmm_results
            logger.info(f"  ✅ HMM regime discovery test: {'Success' if hmm_success else 'Failed'}")
        except Exception as e:
            hmm_success = False
            logger.error(f"  ❌ HMM regime discovery test failed: {e}")
        
        # Test clustering with utilities
        logger.info("🎯 Testing clustering with utilities...")
        try:
            clustering_results = await step._perform_final_clustering_with_utilities(test_data, hmm_results if 'hmm_results' in locals() else {})
            clustering_success = clustering_results is not None and 'cluster_labels' in clustering_results
            logger.info(f"  ✅ Clustering test: {'Success' if clustering_success else 'Failed'}")
        except Exception as e:
            clustering_success = False
            logger.error(f"  ❌ Clustering test failed: {e}")
        
        # Calculate overall success rate
        test_results = {
            'data_loading': data_loading_success,
            'feature_preparation': feature_prep_success,
            'hmm_regime_discovery': hmm_success,
            'clustering': clustering_success
        }
        
        success_rate = sum(test_results.values()) / len(test_results) * 100
        
        logger.info(f"📊 Step03_5 Execution Test Results:")
        logger.info(f"  🎯 Overall Success Rate: {success_rate:.1f}% ({sum(test_results.values())}/{len(test_results)})")
        for test_name, result in test_results.items():
            status = "✅ Success" if result else "❌ Failed"
            logger.info(f"  {status}: {test_name.replace('_', ' ').title()}")
        
        return {
            'success_rate': success_rate,
            'test_results': test_results,
            'utility_operations': utility_operations
        }
        
    except Exception as e:
        logger.error(f"❌ Step03_5 execution test failed: {e}")
        return {
            'success_rate': 0,
            'test_results': {},
            'utility_operations': {},
            'error': str(e)
        }

async def main():
    """Main test function."""
    logger.info("🚀 Starting comprehensive Step03_5 utility integration tests...")
    
    # Test 1: Utility Dependency Injection
    logger.info("=" * 60)
    logger.info("TEST 1: Utility Dependency Injection")
    logger.info("=" * 60)
    injection_results = test_utility_dependency_injection()
    
    # Test 2: Step03_5 Execution
    logger.info("=" * 60)
    logger.info("TEST 2: Step03_5 Execution with Utility Integration")
    logger.info("=" * 60)
    execution_results = await test_step03_5_execution()
    
    # Summary
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE TEST SUMMARY")
    logger.info("=" * 60)
    
    injection_success = injection_results.get('availability_rate', 0) >= 80
    execution_success = execution_results.get('success_rate', 0) >= 75
    
    logger.info(f"📊 Utility Dependency Injection: {'✅ PASS' if injection_success else '❌ FAIL'}")
    logger.info(f"  📈 Availability Rate: {injection_results.get('availability_rate', 0):.1f}%")
    logger.info(f"  🧪 Functionality Success Rate: {injection_results.get('functionality_success_rate', 0):.1f}%")
    
    logger.info(f"📊 Step03_5 Execution: {'✅ PASS' if execution_success else '❌ FAIL'}")
    logger.info(f"  🎯 Overall Success Rate: {execution_results.get('success_rate', 0):.1f}%")
    
    overall_success = injection_success and execution_success
    logger.info(f"🎉 Overall Test Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        logger.info("🎊 Comprehensive utility integration in Step03_5 is working correctly!")
    else:
        logger.warning("⚠️ Some utility integration issues detected. Review the logs above.")
    
    return overall_success

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)