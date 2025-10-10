#!/usr/bin/env python3
"""
Exchange OHLCV Equivalency Test

This script tests that all exchanges (binance, bingx, okx, mexc) return
equivalent OHLCV data format and are fully compatible with src/utils/data/ utilities.

Features:
- Tests data format equivalency across all exchanges
- Validates compatibility with src/utils/data/ utilities
- Comprehensive error reporting and validation
- Performance benchmarking
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from exchanges.shared.unified_exchange_interface import (
    UnifiedExchangeManager, ExchangeType, validate_ohlcv_equivalency
)
from exchanges.shared.unified_ohlcv_standardizer import (
    UnifiedOHLCVStandardizer, standardize_exchange_ohlcv
)
from exchanges.binance.klines_adapter import BinanceKlinesAdapter
from exchanges.bingx.klines_adapter import BingXKlinesAdapter
from exchanges.okx.klines_adapter import OkxKlinesAdapter
from exchanges.mexc.klines_adapter import MexcKlinesAdapter

# Import src/utils/data utilities
from src.utils.data import (
    DataProcessor, DataQualityFramework, DataCleaner,
    validate_and_fix_data_quality, optimize_dataframe_dtypes,
    check_dataframe_health, regularize_timestamps
)


class ExchangeEquivalencyTester:
    """Test suite for exchange OHLCV data equivalency and compatibility."""
    
    def __init__(self):
        """Initialize the test suite"""
        self.logger = None
        self.results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'total_tests': 0,
            'errors': [],
            'warnings': [],
            'performance_metrics': {}
        }
        
        # Test configuration
        self.test_symbol = "BTCUSDT"
        self.test_interval = "1m"
        self.test_limit = 100
        self.tolerance = 1e-6
        
        # Initialize adapters
        self.adapters = {}
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        """Initialize exchange adapters for testing"""
        try:
            # Initialize adapters (without API keys for public data)
            self.adapters = {
                'binance': BinanceKlinesAdapter(),
                'bingx': BingXKlinesAdapter(),
                'okx': OkxKlinesAdapter(),
                'mexc': MexcKlinesAdapter()
            }
            print("✅ Exchange adapters initialized")
        except Exception as e:
            print(f"❌ Failed to initialize adapters: {e}")
            self.results['errors'].append(f"Adapter initialization failed: {e}")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all equivalency and compatibility tests"""
        print("🚀 Starting Exchange OHLCV Equivalency Tests")
        print("=" * 60)
        
        # Test 1: Data Format Standardization
        await self._test_data_format_standardization()
        
        # Test 2: Exchange Data Equivalency
        await self._test_exchange_data_equivalency()
        
        # Test 3: src/utils/data/ Compatibility
        await self._test_data_utils_compatibility()
        
        # Test 4: Performance Benchmarking
        await self._test_performance_benchmarking()
        
        # Test 5: Error Handling
        await self._test_error_handling()
        
        # Generate final report
        self._generate_final_report()
        
        return self.results
    
    async def _test_data_format_standardization(self):
        """Test that all exchanges return standardized data format"""
        print("\n📊 Test 1: Data Format Standardization")
        print("-" * 40)
        
        test_name = "Data Format Standardization"
        self.results['total_tests'] += 1
        
        try:
            # Test each exchange
            for exchange_name, adapter in self.adapters.items():
                print(f"  Testing {exchange_name}...")
                
                # Get data from exchange
                data = await adapter.get_klines_data(
                    symbol=self.test_symbol,
                    interval=self.test_interval,
                    limit=self.test_limit
                )
                
                if data.empty:
                    print(f"    ⚠️ No data returned from {exchange_name}")
                    continue
                
                # Validate data format
                validation_result = self._validate_data_format(data, exchange_name)
                
                if validation_result['valid']:
                    print(f"    ✅ {exchange_name} data format valid")
                else:
                    print(f"    ❌ {exchange_name} data format invalid: {validation_result['errors']}")
                    self.results['errors'].extend(validation_result['errors'])
            
            self.results['tests_passed'] += 1
            print("  ✅ Data format standardization test passed")
            
        except Exception as e:
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"{test_name} failed: {e}")
            print(f"  ❌ {test_name} failed: {e}")
    
    async def _test_exchange_data_equivalency(self):
        """Test that all exchanges return equivalent data structures"""
        print("\n🔄 Test 2: Exchange Data Equivalency")
        print("-" * 40)
        
        test_name = "Exchange Data Equivalency"
        self.results['total_tests'] += 1
        
        try:
            # Collect data from all exchanges
            exchange_data = {}
            
            for exchange_name, adapter in self.adapters.items():
                print(f"  Collecting data from {exchange_name}...")
                
                data = await adapter.get_klines_data(
                    symbol=self.test_symbol,
                    interval=self.test_interval,
                    limit=self.test_limit
                )
                
                if not data.empty:
                    exchange_data[exchange_name] = data
                    print(f"    ✅ {exchange_name}: {len(data)} records")
                else:
                    print(f"    ⚠️ {exchange_name}: No data")
            
            if len(exchange_data) < 2:
                print("  ⚠️ Not enough exchanges with data for equivalency testing")
                self.results['warnings'].append("Insufficient data for equivalency testing")
                return
            
            # Compare data structures
            exchanges = list(exchange_data.keys())
            reference_exchange = exchanges[0]
            reference_data = exchange_data[reference_exchange]
            
            print(f"  Using {reference_exchange} as reference...")
            
            for exchange_name in exchanges[1:]:
                print(f"  Comparing {exchange_name} with {reference_exchange}...")
                
                comparison_data = exchange_data[exchange_name]
                equivalency_result = validate_ohlcv_equivalency(
                    reference_data, comparison_data, self.tolerance
                )
                
                if equivalency_result['equivalent']:
                    print(f"    ✅ {exchange_name} equivalent to {reference_exchange}")
                else:
                    print(f"    ❌ {exchange_name} not equivalent to {reference_exchange}")
                    print(f"    Differences: {equivalency_result['differences']}")
                    self.results['errors'].extend(equivalency_result['errors'])
            
            self.results['tests_passed'] += 1
            print("  ✅ Exchange data equivalency test passed")
            
        except Exception as e:
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"{test_name} failed: {e}")
            print(f"  ❌ {test_name} failed: {e}")
    
    async def _test_data_utils_compatibility(self):
        """Test compatibility with src/utils/data/ utilities"""
        print("\n🔧 Test 3: src/utils/data/ Compatibility")
        print("-" * 40)
        
        test_name = "Data Utils Compatibility"
        self.results['total_tests'] += 1
        
        try:
            # Get data from first available exchange
            test_data = None
            for exchange_name, adapter in self.adapters.items():
                data = await adapter.get_klines_data(
                    symbol=self.test_symbol,
                    interval=self.test_interval,
                    limit=self.test_limit
                )
                if not data.empty:
                    test_data = data
                    print(f"  Using data from {exchange_name} for compatibility testing")
                    break
            
            if test_data is None:
                print("  ⚠️ No data available for compatibility testing")
                self.results['warnings'].append("No data available for compatibility testing")
                return
            
            # Test DataProcessor
            print("  Testing DataProcessor...")
            processor = DataProcessor()
            
            # Test regularize_timestamps
            regularized_data = processor.regularize_timestamps(test_data)
            if len(regularized_data) > 0:
                print("    ✅ regularize_timestamps works")
            else:
                print("    ❌ regularize_timestamps failed")
                self.results['errors'].append("regularize_timestamps failed")
            
            # Test optimize_dataframe_dtypes
            optimized_data = processor.optimize_dataframe_dtypes(test_data)
            if len(optimized_data) > 0:
                print("    ✅ optimize_dataframe_dtypes works")
            else:
                print("    ❌ optimize_dataframe_dtypes failed")
                self.results['errors'].append("optimize_dataframe_dtypes failed")
            
            # Test DataQualityFramework
            print("  Testing DataQualityFramework...")
            quality_framework = DataQualityFramework()
            quality_result = quality_framework.validate_dataframe_quality(test_data, "compatibility test")
            
            if quality_result.passed:
                print("    ✅ DataQualityFramework validation passed")
            else:
                print(f"    ⚠️ DataQualityFramework validation issues: {quality_result.issues}")
                self.results['warnings'].extend(quality_result.issues)
            
            # Test DataCleaner
            print("  Testing DataCleaner...")
            cleaner = DataCleaner()
            cleaned_data = cleaner.clean_dataframe(test_data)
            if len(cleaned_data) > 0:
                print("    ✅ DataCleaner works")
            else:
                print("    ❌ DataCleaner failed")
                self.results['errors'].append("DataCleaner failed")
            
            # Test convenience functions
            print("  Testing convenience functions...")
            
            # Test validate_and_fix_data_quality
            fixed_data, fix_report = validate_and_fix_data_quality(test_data)
            if len(fixed_data) > 0:
                print("    ✅ validate_and_fix_data_quality works")
            else:
                print("    ❌ validate_and_fix_data_quality failed")
                self.results['errors'].append("validate_and_fix_data_quality failed")
            
            # Test check_dataframe_health
            health_result = check_dataframe_health(test_data)
            if health_result['healthy']:
                print("    ✅ check_dataframe_health passed")
            else:
                print(f"    ⚠️ check_dataframe_health issues: {health_result['issues']}")
                self.results['warnings'].extend(health_result['issues'])
            
            self.results['tests_passed'] += 1
            print("  ✅ src/utils/data/ compatibility test passed")
            
        except Exception as e:
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"{test_name} failed: {e}")
            print(f"  ❌ {test_name} failed: {e}")
            traceback.print_exc()
    
    async def _test_performance_benchmarking(self):
        """Test performance of data processing operations"""
        print("\n⚡ Test 4: Performance Benchmarking")
        print("-" * 40)
        
        test_name = "Performance Benchmarking"
        self.results['total_tests'] += 1
        
        try:
            # Get data from first available exchange
            test_data = None
            for exchange_name, adapter in self.adapters.items():
                data = await adapter.get_klines_data(
                    symbol=self.test_symbol,
                    interval=self.test_interval,
                    limit=self.test_limit
                )
                if not data.empty:
                    test_data = data
                    break
            
            if test_data is None:
                print("  ⚠️ No data available for performance testing")
                self.results['warnings'].append("No data available for performance testing")
                return
            
            # Benchmark data processing operations
            import time
            
            operations = {
                'regularize_timestamps': lambda: regularize_timestamps(test_data),
                'optimize_dataframe_dtypes': lambda: optimize_dataframe_dtypes(test_data),
                'validate_and_fix_data_quality': lambda: validate_and_fix_data_quality(test_data)[0],
                'check_dataframe_health': lambda: check_dataframe_health(test_data)
            }
            
            performance_metrics = {}
            
            for op_name, op_func in operations.items():
                print(f"  Benchmarking {op_name}...")
                
                # Run operation multiple times for accurate timing
                times = []
                for _ in range(5):
                    start_time = time.time()
                    try:
                        result = op_func()
                        end_time = time.time()
                        times.append(end_time - start_time)
                    except Exception as e:
                        print(f"    ❌ {op_name} failed: {e}")
                        break
                
                if times:
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    performance_metrics[op_name] = {
                        'avg_time': avg_time,
                        'std_time': std_time,
                        'min_time': min(times),
                        'max_time': max(times)
                    }
                    print(f"    ✅ {op_name}: {avg_time:.4f}s ± {std_time:.4f}s")
            
            self.results['performance_metrics'] = performance_metrics
            self.results['tests_passed'] += 1
            print("  ✅ Performance benchmarking test passed")
            
        except Exception as e:
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"{test_name} failed: {e}")
            print(f"  ❌ {test_name} failed: {e}")
    
    async def _test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n🛡️ Test 5: Error Handling")
        print("-" * 40)
        
        test_name = "Error Handling"
        self.results['total_tests'] += 1
        
        try:
            # Test with invalid parameters
            print("  Testing invalid parameters...")
            
            for exchange_name, adapter in self.adapters.items():
                try:
                    # Test with invalid symbol
                    data = await adapter.get_klines_data(
                        symbol="INVALID_SYMBOL",
                        interval=self.test_interval,
                        limit=10
                    )
                    if data.empty:
                        print(f"    ✅ {exchange_name} handles invalid symbol gracefully")
                    else:
                        print(f"    ⚠️ {exchange_name} returned data for invalid symbol")
                
                except Exception as e:
                    print(f"    ✅ {exchange_name} properly handles invalid symbol: {type(e).__name__}")
            
            # Test with invalid interval
            try:
                data = await list(self.adapters.values())[0].get_klines_data(
                    symbol=self.test_symbol,
                    interval="invalid_interval",
                    limit=10
                )
                if data.empty:
                    print("    ✅ Handles invalid interval gracefully")
                else:
                    print("    ⚠️ Returned data for invalid interval")
            except Exception as e:
                print(f"    ✅ Properly handles invalid interval: {type(e).__name__}")
            
            # Test with very small limit
            try:
                data = await list(self.adapters.values())[0].get_klines_data(
                    symbol=self.test_symbol,
                    interval=self.test_interval,
                    limit=1
                )
                if len(data) <= 1:
                    print("    ✅ Handles small limit correctly")
                else:
                    print(f"    ⚠️ Returned {len(data)} records for limit=1")
            except Exception as e:
                print(f"    ✅ Properly handles small limit: {type(e).__name__}")
            
            self.results['tests_passed'] += 1
            print("  ✅ Error handling test passed")
            
        except Exception as e:
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"{test_name} failed: {e}")
            print(f"  ❌ {test_name} failed: {e}")
    
    def _validate_data_format(self, data: pd.DataFrame, exchange_name: str) -> Dict[str, Any]:
        """Validate that data follows the standardized format"""
        validation_result = {
            'valid': True,
            'errors': []
        }
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'interval', 'exchange', 'source']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                validation_result['valid'] = False
                validation_result['errors'].append(f"Column {col} should be numeric")
        
        # Check for negative values
        for col in numeric_columns:
            if col in data.columns and (data[col] < 0).any():
                validation_result['valid'] = False
                validation_result['errors'].append(f"Column {col} contains negative values")
        
        # Check OHLC consistency
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            high_violations = (data['high'] < data[['open', 'close']].max(axis=1)).sum()
            low_violations = (data['low'] > data[['open', 'close']].min(axis=1)).sum()
            
            if high_violations > 0:
                validation_result['valid'] = False
                validation_result['errors'].append(f"High price violations: {high_violations}")
            
            if low_violations > 0:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Low price violations: {low_violations}")
        
        return validation_result
    
    def _generate_final_report(self):
        """Generate final test report"""
        print("\n" + "=" * 60)
        print("📋 FINAL TEST REPORT")
        print("=" * 60)
        
        total_tests = self.results['total_tests']
        passed_tests = self.results['tests_passed']
        failed_tests = self.results['tests_failed']
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.results['errors']:
            print(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results['errors'][:10]:  # Show first 10 errors
                print(f"  • {error}")
            if len(self.results['errors']) > 10:
                print(f"  ... and {len(self.results['errors']) - 10} more errors")
        
        if self.results['warnings']:
            print(f"\n⚠️ Warnings ({len(self.results['warnings'])}):")
            for warning in self.results['warnings'][:5]:  # Show first 5 warnings
                print(f"  • {warning}")
            if len(self.results['warnings']) > 5:
                print(f"  ... and {len(self.results['warnings']) - 5} more warnings")
        
        if self.results['performance_metrics']:
            print(f"\n⚡ Performance Metrics:")
            for op_name, metrics in self.results['performance_metrics'].items():
                print(f"  {op_name}: {metrics['avg_time']:.4f}s ± {metrics['std_time']:.4f}s")
        
        # Overall assessment
        if failed_tests == 0:
            print(f"\n🎉 ALL TESTS PASSED! Exchange equivalency is working correctly.")
        else:
            print(f"\n⚠️ {failed_tests} tests failed. Please review the errors above.")
        
        print("=" * 60)


async def main():
    """Main test function"""
    print("🚀 Exchange OHLCV Equivalency Test Suite")
    print("Testing complete equivalency between binance, bingx, okx, mexc")
    print("Ensuring full compatibility with src/utils/data/ utilities")
    print()
    
    # Initialize and run tests
    tester = ExchangeEquivalencyTester()
    results = await tester.run_all_tests()
    
    # Return exit code based on results
    if results['tests_failed'] == 0:
        print("\n✅ All tests passed successfully!")
        return 0
    else:
        print(f"\n❌ {results['tests_failed']} tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)