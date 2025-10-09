"""
Comprehensive Tests for Timeframe Optimization

This module tests the timeframe-aware feature lookback optimization
across 5m, 15m, and 60m timeframes to ensure optimal performance.
"""

import asyncio
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)

from .timeframe_aware_optimizer import TimeframeAwareFeatureLookbackOptimizer
from .timeframe_config_loader import get_timeframe_config_loader, validate_timeframe


class TimeframeOptimizationTester:
    """Comprehensive tester for timeframe optimization."""
    
    def __init__(self):
        """Initialize the tester."""
        self.optimizer = TimeframeAwareFeatureLookbackOptimizer()
        self.config_loader = get_timeframe_config_loader()
        self.test_results = {}
        
        tprint("🧪 Initializing Timeframe Optimization Tester")
    
    def generate_test_data(self, timeframe: str, n_samples: int = 1000) -> pd.DataFrame:
        """Generate test data for a specific timeframe."""
        # Convert timeframe to minutes
        timeframe_minutes = {
            '5m': 5,
            '15m': 15,
            '60m': 60
        }.get(timeframe, 15)
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=30)
        timestamps = [start_time + timedelta(minutes=timeframe_minutes*i) for i in range(n_samples)]
        
        # Generate synthetic OHLCV data
        np.random.seed(42)
        base_price = 100.0
        returns = np.random.normal(0, 0.02, n_samples)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 5000, n_samples)
        })
        
        data.set_index('timestamp', inplace=True)
        
        return data
    
    def create_pipeline_state(self, timeframe: str, symbol: str = 'ETHUSDT') -> Dict[str, Any]:
        """Create pipeline state for testing."""
        return {
            'symbol': symbol,
            'exchange': 'binance',
            'timeframe': timeframe,
            'data_dir': 'test_data',
            'custom_params': {
                'enable_matrix_optimization': True,
                'enable_hardware_optimization': True,
                'verbose_logging': True
            }
        }
    
    async def test_timeframe_configuration(self, timeframe: str) -> Dict[str, Any]:
        """Test configuration for a specific timeframe."""
        tprint(f"\n🔧 Testing {timeframe.upper()} Configuration")
        
        start_time = time.time()
        
        try:
            # Validate configuration
            is_valid = validate_timeframe(timeframe)
            if not is_valid:
                return {
                    'timeframe': timeframe,
                    'success': False,
                    'error': 'Configuration validation failed',
                    'execution_time': time.time() - start_time
                }
            
            # Get configuration info
            config_info = self.optimizer.get_timeframe_info(timeframe)
            
            # Test configuration loading
            config_loader = get_timeframe_config_loader()
            config = config_loader.get_timeframe_config(timeframe)
            
            execution_time = time.time() - start_time
            
            tprint_success(f"✅ {timeframe.upper()} configuration test passed")
            
            return {
                'timeframe': timeframe,
                'success': True,
                'config_info': config_info,
                'base_period_minutes': config.base_period_minutes if config else None,
                'lookback_range': f"{config.min_lookback}-{config.max_lookback}" if config else None,
                'execution_time': execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Configuration test failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            
            return {
                'timeframe': timeframe,
                'success': False,
                'error': error_msg,
                'execution_time': execution_time
            }
    
    async def test_optimization_performance(self, timeframe: str) -> Dict[str, Any]:
        """Test optimization performance for a specific timeframe."""
        tprint(f"\n⚡ Testing {timeframe.upper()} Optimization Performance")
        
        start_time = time.time()
        
        try:
            # Generate test data
            test_data = self.generate_test_data(timeframe, n_samples=500)  # Smaller for testing
            
            # Create pipeline state
            pipeline_state = self.create_pipeline_state(timeframe)
            
            # Add test data to pipeline state
            pipeline_state['market_data'] = test_data
            
            # Execute optimization
            result = await self.optimizer.execute(None, pipeline_state)
            
            execution_time = time.time() - start_time
            
            if result.get('success', False):
                tprint_success(f"✅ {timeframe.upper()} optimization test passed")
                
                return {
                    'timeframe': timeframe,
                    'success': True,
                    'execution_time': execution_time,
                    'optimization_time': result.get('execution_time', 0),
                    'features_optimized': result.get('features_optimized', 0),
                    'best_ic': result.get('best_ic', 0),
                    'error': None
                }
            else:
                error_msg = result.get('error_message', 'Unknown error')
                tprint_error(f"❌ {timeframe.upper()} optimization test failed: {error_msg}")
                
                return {
                    'timeframe': timeframe,
                    'success': False,
                    'execution_time': execution_time,
                    'optimization_time': 0,
                    'features_optimized': 0,
                    'best_ic': 0,
                    'error': error_msg
                }
                
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Performance test failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            
            return {
                'timeframe': timeframe,
                'success': False,
                'execution_time': execution_time,
                'optimization_time': 0,
                'features_optimized': 0,
                'best_ic': 0,
                'error': error_msg
            }
    
    async def test_all_timeframes(self) -> Dict[str, Any]:
        """Test all supported timeframes."""
        tprint("🚀 Starting Comprehensive Timeframe Testing")
        
        timeframes = ['5m', '15m', '60m']
        results = {
            'configuration_tests': {},
            'performance_tests': {},
            'summary': {}
        }
        
        # Test configurations
        tprint("\n📋 Phase 1: Configuration Testing")
        for timeframe in timeframes:
            results['configuration_tests'][timeframe] = await self.test_timeframe_configuration(timeframe)
        
        # Test performance
        tprint("\n⚡ Phase 2: Performance Testing")
        for timeframe in timeframes:
            results['performance_tests'][timeframe] = await self.test_optimization_performance(timeframe)
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary."""
        config_tests = results['configuration_tests']
        perf_tests = results['performance_tests']
        
        summary = {
            'total_timeframes_tested': len(config_tests),
            'configuration_success_rate': 0,
            'performance_success_rate': 0,
            'average_execution_time': 0,
            'timeframe_rankings': [],
            'recommendations': []
        }
        
        # Calculate success rates
        config_successes = sum(1 for test in config_tests.values() if test['success'])
        perf_successes = sum(1 for test in perf_tests.values() if test['success'])
        
        summary['configuration_success_rate'] = config_successes / len(config_tests) * 100
        summary['performance_success_rate'] = perf_successes / len(perf_tests) * 100
        
        # Calculate average execution time
        execution_times = [test['execution_time'] for test in perf_tests.values() if test['success']]
        if execution_times:
            summary['average_execution_time'] = sum(execution_times) / len(execution_times)
        
        # Rank timeframes by performance
        timeframe_scores = []
        for timeframe in config_tests.keys():
            config_success = config_tests[timeframe]['success']
            perf_success = perf_tests[timeframe]['success']
            exec_time = perf_tests[timeframe]['execution_time']
            
            # Score based on success and speed
            score = 0
            if config_success:
                score += 50
            if perf_success:
                score += 50
            if exec_time > 0:
                score += max(0, 100 - exec_time)  # Faster is better
            
            timeframe_scores.append({
                'timeframe': timeframe,
                'score': score,
                'config_success': config_success,
                'perf_success': perf_success,
                'execution_time': exec_time
            })
        
        summary['timeframe_rankings'] = sorted(timeframe_scores, key=lambda x: x['score'], reverse=True)
        
        # Generate recommendations
        recommendations = []
        
        if summary['configuration_success_rate'] < 100:
            recommendations.append("Some timeframe configurations need attention")
        
        if summary['performance_success_rate'] < 100:
            recommendations.append("Some timeframe optimizations are failing")
        
        if summary['average_execution_time'] > 300:  # 5 minutes
            recommendations.append("Consider optimizing execution time for better performance")
        
        # Find best performing timeframe
        if summary['timeframe_rankings']:
            best_timeframe = summary['timeframe_rankings'][0]
            recommendations.append(f"Best performing timeframe: {best_timeframe['timeframe']} (score: {best_timeframe['score']:.1f})")
        
        summary['recommendations'] = recommendations
        
        return summary
    
    def print_test_results(self, results: Dict[str, Any]):
        """Print comprehensive test results."""
        tprint("\n📊 COMPREHENSIVE TEST RESULTS")
        tprint("=" * 50)
        
        # Configuration test results
        tprint("\n📋 Configuration Test Results:")
        for timeframe, test in results['configuration_tests'].items():
            status = "✅ PASS" if test['success'] else "❌ FAIL"
            tprint(f"   {timeframe.upper()}: {status}")
            if test['success'] and 'lookback_range' in test:
                tprint(f"      → Lookback range: {test['lookback_range']}")
                tprint(f"      → Base period: {test['base_period_minutes']} minutes")
            elif not test['success']:
                tprint(f"      → Error: {test['error']}")
        
        # Performance test results
        tprint("\n⚡ Performance Test Results:")
        for timeframe, test in results['performance_tests'].items():
            status = "✅ PASS" if test['success'] else "❌ FAIL"
            tprint(f"   {timeframe.upper()}: {status}")
            if test['success']:
                tprint(f"      → Execution time: {test['execution_time']:.3f}s")
                tprint(f"      → Features optimized: {test['features_optimized']}")
                tprint(f"      → Best IC: {test['best_ic']:.4f}")
            else:
                tprint(f"      → Error: {test['error']}")
        
        # Summary
        summary = results['summary']
        tprint(f"\n📈 Summary:")
        tprint(f"   → Timeframes tested: {summary['total_timeframes_tested']}")
        tprint(f"   → Configuration success rate: {summary['configuration_success_rate']:.1f}%")
        tprint(f"   → Performance success rate: {summary['performance_success_rate']:.1f}%")
        tprint(f"   → Average execution time: {summary['average_execution_time']:.3f}s")
        
        # Rankings
        tprint(f"\n🏆 Timeframe Rankings:")
        for i, ranking in enumerate(summary['timeframe_rankings'], 1):
            tprint(f"   {i}. {ranking['timeframe'].upper()} (score: {ranking['score']:.1f})")
        
        # Recommendations
        if summary['recommendations']:
            tprint(f"\n💡 Recommendations:")
            for rec in summary['recommendations']:
                tprint(f"   → {rec}")
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'optimizer'):
            self.optimizer.cleanup()


async def run_comprehensive_tests():
    """Run comprehensive timeframe optimization tests."""
    tester = TimeframeOptimizationTester()
    
    try:
        # Run all tests
        results = await tester.test_all_timeframes()
        
        # Print results
        tester.print_test_results(results)
        
        # Return results for further analysis
        return results
        
    finally:
        tester.cleanup()


if __name__ == "__main__":
    # Run comprehensive tests
    asyncio.run(run_comprehensive_tests())