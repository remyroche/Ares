#!/usr/bin/env python3
"""
Exchange OHLCV Data Equivalency Test

This script tests the complete equivalency between exchanges (binance, bingx, okx, mexc)
and validates that the OHLCV data we get from them through ExchangeInterface is fully
standardized for downstream use and compatible with src/utils/data/.

Usage:
    python test_exchange_equivalency.py
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from exchanges.shared.enhanced_unified_exchange_interface import (
    EnhancedUnifiedExchangeManager, ExchangeType, DataQualityLevel
)
from exchanges.shared.unified_exchange_standardizer import (
    UnifiedExchangeStandardizer, standardize_exchange_ohlcv, validate_ohlcv_equivalency
)
from exchanges.binance.klines_adapter import BinanceKlinesAdapter
from exchanges.bingx.klines_adapter import BingXKlinesAdapter
from exchanges.okx.klines_adapter import OkxKlinesAdapter
from exchanges.mexc.klines_adapter import MexcKlinesAdapter


class ExchangeEquivalencyTester:
    """Test suite for validating exchange data equivalency"""
    
    def __init__(self):
        """Initialize the equivalency tester"""
        self.logger = system_logger.getChild("ExchangeEquivalencyTester")
        self.manager = EnhancedUnifiedExchangeManager(DataQualityLevel.STANDARD)
        self.standardizer = UnifiedExchangeStandardizer(DataQualityLevel.STANDARD)
        
        # Test configuration
        self.test_symbol = "BTCUSDT"
        self.test_interval = "1m"
        self.test_limit = 100
        
        # Initialize adapters
        self.adapters = {}
        self._initialize_adapters()
        
        self.logger.info("✅ ExchangeEquivalencyTester initialized")
    
    def _initialize_adapters(self):
        """Initialize exchange adapters"""
        try:
            # Initialize adapters (without API keys for public data)
            self.adapters[ExchangeType.BINANCE] = BinanceKlinesAdapter()
            self.adapters[ExchangeType.BINGX] = BingXKlinesAdapter()
            self.adapters[ExchangeType.OKX] = OkxKlinesAdapter()
            self.adapters[ExchangeType.MEXC] = MexcKlinesAdapter()
            
            # Register with manager
            for exchange_type, adapter in self.adapters.items():
                if adapter.unified_adapter:
                    self.manager.register_exchange(
                        adapter.unified_adapter.exchange_instance,
                        exchange_type,
                        DataQualityLevel.STANDARD
                    )
            
            self.logger.info(f"✅ Initialized {len(self.adapters)} exchange adapters")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize adapters: {e}")
    
    async def test_data_standardization(self) -> Dict[str, Any]:
        """Test that all exchanges produce standardized data"""
        self.logger.info("🧪 Testing data standardization...")
        
        results = {
            'test_name': 'data_standardization',
            'exchanges_tested': [],
            'standardization_results': {},
            'overall_success': True,
            'issues': []
        }
        
        try:
            for exchange_type, adapter in self.adapters.items():
                if not adapter.unified_adapter:
                    continue
                
                self.logger.info(f"Testing {exchange_type.value} standardization...")
                
                try:
                    # Get standardized data
                    df = await adapter.get_klines_data(
                        self.test_symbol, 
                        self.test_interval, 
                        limit=self.test_limit
                    )
                    
                    if df.empty:
                        results['issues'].append(f"{exchange_type.value}: No data returned")
                        continue
                    
                    # Validate standardization
                    standardization_result = self._validate_standardization(df, exchange_type)
                    results['standardization_results'][exchange_type.value] = standardization_result
                    results['exchanges_tested'].append(exchange_type.value)
                    
                    if not standardization_result['valid']:
                        results['overall_success'] = False
                        results['issues'].extend(standardization_result['errors'])
                    
                except Exception as e:
                    results['issues'].append(f"{exchange_type.value}: {str(e)}")
                    results['overall_success'] = False
            
            return results
            
        except Exception as e:
            results['overall_success'] = False
            results['issues'].append(f"Standardization test failed: {str(e)}")
            return results
    
    def _validate_standardization(self, df: pd.DataFrame, exchange_type: ExchangeType) -> Dict[str, Any]:
        """Validate that data is properly standardized"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'required_columns_present': True,
            'timestamp_index': False,
            'numeric_ohlcv': True
        }
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            result['required_columns_present'] = False
            result['errors'].append(f"Missing required columns: {missing_columns}")
            result['valid'] = False
        
        # Check timestamp index
        if isinstance(df.index, pd.DatetimeIndex):
            result['timestamp_index'] = True
        else:
            result['warnings'].append("Timestamp is not set as index")
        
        # Check OHLCV data types
        for col in required_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    result['numeric_ohlcv'] = False
                    result['errors'].append(f"Column {col} is not numeric: {df[col].dtype}")
                    result['valid'] = False
        
        # Check for exchange metadata
        if 'exchange' in df.columns:
            unique_exchanges = df['exchange'].unique()
            if len(unique_exchanges) == 1 and unique_exchanges[0] == exchange_type.value:
                result['exchange_metadata_correct'] = True
            else:
                result['warnings'].append(f"Exchange metadata mismatch: {unique_exchanges}")
        else:
            result['warnings'].append("Exchange metadata column missing")
        
        return result
    
    async def test_data_equivalency(self) -> Dict[str, Any]:
        """Test that data from different exchanges is equivalent"""
        self.logger.info("🧪 Testing data equivalency...")
        
        results = {
            'test_name': 'data_equivalency',
            'exchanges_compared': [],
            'equivalency_results': {},
            'overall_equivalent': True,
            'issues': []
        }
        
        try:
            # Get data from all exchanges
            all_data = {}
            for exchange_type, adapter in self.adapters.items():
                if not adapter.unified_adapter:
                    continue
                
                try:
                    df = await adapter.get_klines_data(
                        self.test_symbol, 
                        self.test_interval, 
                        limit=self.test_limit
                    )
                    if not df.empty:
                        all_data[exchange_type] = df
                        results['exchanges_compared'].append(exchange_type.value)
                except Exception as e:
                    results['issues'].append(f"Failed to get data from {exchange_type.value}: {str(e)}")
            
            if len(all_data) < 2:
                results['issues'].append("Need at least 2 exchanges with data for equivalency test")
                results['overall_equivalent'] = False
                return results
            
            # Compare each pair of exchanges
            exchange_list = list(all_data.keys())
            for i, exchange1 in enumerate(exchange_list):
                for exchange2 in exchange_list[i+1:]:
                    pair_key = f"{exchange1.value}_vs_{exchange2.value}"
                    
                    try:
                        equivalency_result = validate_ohlcv_equivalency(
                            all_data[exchange1], all_data[exchange2]
                        )
                        results['equivalency_results'][pair_key] = equivalency_result
                        
                        if not equivalency_result['equivalent']:
                            results['overall_equivalent'] = False
                            results['issues'].extend(equivalency_result['errors'])
                    
                    except Exception as e:
                        results['issues'].append(f"Equivalency test failed for {pair_key}: {str(e)}")
                        results['overall_equivalent'] = False
            
            return results
            
        except Exception as e:
            results['overall_equivalent'] = False
            results['issues'].append(f"Equivalency test failed: {str(e)}")
            return results
    
    async def test_src_utils_compatibility(self) -> Dict[str, Any]:
        """Test compatibility with src/utils/data/ utilities"""
        self.logger.info("🧪 Testing src/utils/data/ compatibility...")
        
        results = {
            'test_name': 'src_utils_compatibility',
            'exchanges_tested': [],
            'compatibility_results': {},
            'overall_compatible': True,
            'issues': []
        }
        
        try:
            for exchange_type, adapter in self.adapters.items():
                if not adapter.unified_adapter:
                    continue
                
                self.logger.info(f"Testing {exchange_type.value} compatibility...")
                
                try:
                    # Get standardized data
                    df = await adapter.get_klines_data(
                        self.test_symbol, 
                        self.test_interval, 
                        limit=self.test_limit
                    )
                    
                    if df.empty:
                        results['issues'].append(f"{exchange_type.value}: No data for compatibility test")
                        continue
                    
                    # Test compatibility with src/utils/data/ utilities
                    compatibility_result = self._test_data_utils_compatibility(df, exchange_type)
                    results['compatibility_results'][exchange_type.value] = compatibility_result
                    results['exchanges_tested'].append(exchange_type.value)
                    
                    if not compatibility_result['compatible']:
                        results['overall_compatible'] = False
                        results['issues'].extend(compatibility_result['errors'])
                
                except Exception as e:
                    results['issues'].append(f"{exchange_type.value}: {str(e)}")
                    results['overall_compatible'] = False
            
            return results
            
        except Exception as e:
            results['overall_compatible'] = False
            results['issues'].append(f"Compatibility test failed: {str(e)}")
            return results
    
    def _test_data_utils_compatibility(self, df: pd.DataFrame, exchange_type: ExchangeType) -> Dict[str, Any]:
        """Test compatibility with src/utils/data/ utilities"""
        result = {
            'compatible': True,
            'errors': [],
            'warnings': [],
            'tests_passed': [],
            'tests_failed': []
        }
        
        try:
            # Test 1: DataProcessor.optimize_dataframe_dtypes
            try:
                from src.utils.data import DataProcessor
                processor = DataProcessor()
                optimized_df = processor.optimize_dataframe_dtypes(df)
                result['tests_passed'].append('optimize_dataframe_dtypes')
            except Exception as e:
                result['tests_failed'].append(f'optimize_dataframe_dtypes: {str(e)}')
                result['compatible'] = False
            
            # Test 2: DataProcessor.regularize_timestamps
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    regularized_df = processor.regularize_timestamps(df)
                    result['tests_passed'].append('regularize_timestamps')
                else:
                    result['warnings'].append('Cannot test regularize_timestamps: no datetime index')
            except Exception as e:
                result['tests_failed'].append(f'regularize_timestamps: {str(e)}')
                result['compatible'] = False
            
            # Test 3: DataQualityFramework.validate_dataframe_quality
            try:
                from src.utils.data import DataQualityFramework
                quality_framework = DataQualityFramework()
                quality_result = quality_framework.validate_dataframe_quality(df, f"{exchange_type.value} test")
                result['tests_passed'].append('validate_dataframe_quality')
                result['quality_score'] = quality_result.quality_score
            except Exception as e:
                result['tests_failed'].append(f'validate_dataframe_quality: {str(e)}')
                result['compatible'] = False
            
            # Test 4: DataCleaner.handle_missing_values_intelligently
            try:
                from src.utils.data import DataCleaner
                cleaner = DataCleaner()
                cleaned_df = cleaner.handle_missing_values_intelligently(df)
                result['tests_passed'].append('handle_missing_values_intelligently')
            except Exception as e:
                result['tests_failed'].append(f'handle_missing_values_intelligently: {str(e)}')
                result['compatible'] = False
            
            # Test 5: check_dataframe_health
            try:
                from src.utils.data import check_dataframe_health
                health_result = check_dataframe_health(df)
                result['tests_passed'].append('check_dataframe_health')
                result['health_score'] = health_result.get('overall_health_score', 0)
            except Exception as e:
                result['tests_failed'].append(f'check_dataframe_health: {str(e)}')
                result['compatible'] = False
            
        except Exception as e:
            result['errors'].append(f"Compatibility test failed: {str(e)}")
            result['compatible'] = False
        
        return result
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive equivalency test suite"""
        self.logger.info("🚀 Starting comprehensive exchange equivalency test...")
        
        test_results = {
            'test_suite': 'exchange_equivalency',
            'timestamp': datetime.now().isoformat(),
            'test_config': {
                'symbol': self.test_symbol,
                'interval': self.test_interval,
                'limit': self.test_limit
            },
            'tests': {},
            'overall_success': True,
            'summary': {}
        }
        
        try:
            # Test 1: Data Standardization
            standardization_results = await self.test_data_standardization()
            test_results['tests']['standardization'] = standardization_results
            if not standardization_results['overall_success']:
                test_results['overall_success'] = False
            
            # Test 2: Data Equivalency
            equivalency_results = await self.test_data_equivalency()
            test_results['tests']['equivalency'] = equivalency_results
            if not equivalency_results['overall_equivalent']:
                test_results['overall_success'] = False
            
            # Test 3: src/utils/data/ Compatibility
            compatibility_results = await self.test_src_utils_compatibility()
            test_results['tests']['compatibility'] = compatibility_results
            if not compatibility_results['overall_compatible']:
                test_results['overall_success'] = False
            
            # Generate summary
            test_results['summary'] = self._generate_test_summary(test_results)
            
            return test_results
            
        except Exception as e:
            test_results['overall_success'] = False
            test_results['error'] = str(e)
            self.logger.error(f"Comprehensive test failed: {e}")
            return test_results
    
    def _generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary"""
        summary = {
            'total_tests': len(test_results['tests']),
            'passed_tests': 0,
            'failed_tests': 0,
            'exchanges_tested': set(),
            'total_issues': 0,
            'critical_issues': 0
        }
        
        for test_name, test_result in test_results['tests'].items():
            if test_name == 'standardization':
                if test_result['overall_success']:
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                summary['exchanges_tested'].update(test_result.get('exchanges_tested', []))
                summary['total_issues'] += len(test_result.get('issues', []))
            
            elif test_name == 'equivalency':
                if test_result['overall_equivalent']:
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                summary['exchanges_tested'].update(test_result.get('exchanges_compared', []))
                summary['total_issues'] += len(test_result.get('issues', []))
            
            elif test_name == 'compatibility':
                if test_result['overall_compatible']:
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                summary['exchanges_tested'].update(test_result.get('exchanges_tested', []))
                summary['total_issues'] += len(test_result.get('issues', []))
        
        summary['exchanges_tested'] = list(summary['exchanges_tested'])
        summary['success_rate'] = summary['passed_tests'] / summary['total_tests'] * 100 if summary['total_tests'] > 0 else 0
        
        return summary
    
    def print_test_results(self, test_results: Dict[str, Any]):
        """Print test results in a formatted way"""
        print("\n" + "="*80)
        print("🔬 EXCHANGE OHLCV DATA EQUIVALENCY TEST RESULTS")
        print("="*80)
        
        print(f"📊 Overall Success: {'✅ YES' if test_results['overall_success'] else '❌ NO'}")
        print(f"⏰ Test Time: {test_results['timestamp']}")
        print(f"🎯 Test Symbol: {test_results['test_config']['symbol']}")
        print(f"⏱️  Test Interval: {test_results['test_config']['interval']}")
        print(f"📈 Test Limit: {test_results['test_config']['limit']}")
        
        summary = test_results.get('summary', {})
        print(f"\n📋 SUMMARY:")
        print(f"   Total Tests: {summary.get('total_tests', 0)}")
        print(f"   Passed: {summary.get('passed_tests', 0)}")
        print(f"   Failed: {summary.get('failed_tests', 0)}")
        print(f"   Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"   Exchanges Tested: {', '.join(summary.get('exchanges_tested', []))}")
        print(f"   Total Issues: {summary.get('total_issues', 0)}")
        
        # Print individual test results
        for test_name, test_result in test_results.get('tests', {}).items():
            print(f"\n🧪 {test_name.upper()} TEST:")
            
            if test_name == 'standardization':
                success = test_result.get('overall_success', False)
                print(f"   Status: {'✅ PASSED' if success else '❌ FAILED'}")
                print(f"   Exchanges: {', '.join(test_result.get('exchanges_tested', []))}")
                
                if test_result.get('issues'):
                    print("   Issues:")
                    for issue in test_result['issues'][:5]:  # Show first 5 issues
                        print(f"     • {issue}")
                    if len(test_result['issues']) > 5:
                        print(f"     • ... and {len(test_result['issues']) - 5} more issues")
            
            elif test_name == 'equivalency':
                success = test_result.get('overall_equivalent', False)
                print(f"   Status: {'✅ PASSED' if success else '❌ FAILED'}")
                print(f"   Exchanges: {', '.join(test_result.get('exchanges_compared', []))}")
                
                # Show equivalency results
                for pair, result in test_result.get('equivalency_results', {}).items():
                    status = '✅' if result.get('equivalent', False) else '❌'
                    print(f"     {pair}: {status}")
                
                if test_result.get('issues'):
                    print("   Issues:")
                    for issue in test_result['issues'][:5]:
                        print(f"     • {issue}")
            
            elif test_name == 'compatibility':
                success = test_result.get('overall_compatible', False)
                print(f"   Status: {'✅ PASSED' if success else '❌ FAILED'}")
                print(f"   Exchanges: {', '.join(test_result.get('exchanges_tested', []))}")
                
                # Show compatibility results for each exchange
                for exchange, result in test_result.get('compatibility_results', {}).items():
                    status = '✅' if result.get('compatible', False) else '❌'
                    tests_passed = len(result.get('tests_passed', []))
                    tests_failed = len(result.get('tests_failed', []))
                    print(f"     {exchange}: {status} ({tests_passed} passed, {tests_failed} failed)")
        
        print("\n" + "="*80)


async def main():
    """Main test function"""
    print("🚀 Starting Exchange OHLCV Data Equivalency Test...")
    
    try:
        # Initialize tester
        tester = ExchangeEquivalencyTester()
        
        # Run comprehensive test
        results = await tester.run_comprehensive_test()
        
        # Print results
        tester.print_test_results(results)
        
        # Return success status
        return results['overall_success']
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(main())
    sys.exit(0 if success else 1)