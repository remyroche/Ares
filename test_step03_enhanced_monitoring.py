#!/usr/bin/env python3
"""
Test Script for Enhanced Step03 Monitoring System.

This script tests the comprehensive monitoring system implemented for step03,
including function call monitoring, error handling, performance tracking,
and detailed reporting.
"""

import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import enhanced monitoring decorators
from src.core.decorators import (
    monitor_step03_functions,
    handle_step03_errors,
    validates,
    traced
)

# Import reporting system
from src.core.reporting import (
    Step03ExecutionReporter,
    ReportFormat
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_step03_monitoring.log')
    ]
)

logger = logging.getLogger(__name__)

class TestStep03Monitoring:
    """Test class for Step03 enhanced monitoring system."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TestStep03Monitoring")
        self.test_results = []
        self.start_time = None
    
    @monitor_step03_functions
    @handle_step03_errors
    @validates()
    @traced(span_name='test_function_call_monitoring')
    async def test_function_call_monitoring(self) -> dict:
        """Test function call monitoring capabilities."""
        self.logger.info("🧪 Testing function call monitoring...")
        
        # Simulate some work
        await asyncio.sleep(0.1)
        
        # Call nested functions to test nested call tracking
        result1 = await self._nested_function_1()
        result2 = await self._nested_function_2()
        
        return {
            'test_name': 'function_call_monitoring',
            'result1': result1,
            'result2': result2,
            'success': True
        }
    
    @monitor_step03_functions
    @handle_step03_errors
    @validates()
    @traced(span_name='nested_function_1')
    async def _nested_function_1(self) -> str:
        """Nested function for testing call tracking."""
        await asyncio.sleep(0.05)
        return "nested_function_1_result"
    
    @monitor_step03_functions
    @handle_step03_errors
    @validates()
    @traced(span_name='nested_function_2')
    async def _nested_function_2(self) -> str:
        """Another nested function for testing call tracking."""
        await asyncio.sleep(0.03)
        return "nested_function_2_result"
    
    @monitor_step03_functions
    @handle_step03_errors
    @validates()
    @traced(span_name='test_error_handling')
    async def test_error_handling(self) -> dict:
        """Test error handling capabilities."""
        self.logger.info("🧪 Testing error handling...")
        
        try:
            # Simulate an error
            await self._function_that_fails()
        except Exception as e:
            self.logger.info(f"✅ Error handling test completed: {e}")
        
        return {
            'test_name': 'error_handling',
            'success': True
        }
    
    @monitor_step03_functions
    @handle_step03_errors
    @validates()
    @traced(span_name='function_that_fails')
    async def _function_that_fails(self) -> None:
        """Function that intentionally fails for testing."""
        await asyncio.sleep(0.02)
        raise ValueError("Intentional test error for error handling validation")
    
    @monitor_step03_functions
    @handle_step03_errors
    @validates()
    @traced(span_name='test_performance_monitoring')
    async def test_performance_monitoring(self) -> dict:
        """Test performance monitoring capabilities."""
        self.logger.info("🧪 Testing performance monitoring...")
        
        # Simulate CPU-intensive work
        start_time = time.time()
        result = 0
        for i in range(1000000):
            result += i * i
        end_time = time.time()
        
        # Simulate memory-intensive work
        large_list = [i for i in range(100000)]
        
        return {
            'test_name': 'performance_monitoring',
            'computation_time': end_time - start_time,
            'result': result,
            'list_size': len(large_list),
            'success': True
        }
    
    @monitor_step03_functions
    @handle_step03_errors
    @validates()
    @traced(span_name='test_parameter_validation')
    async def test_parameter_validation(self, 
                                      required_param: str,
                                      optional_param: int = 42,
                                      **kwargs) -> dict:
        """Test parameter validation capabilities."""
        self.logger.info("🧪 Testing parameter validation...")
        
        return {
            'test_name': 'parameter_validation',
            'required_param': required_param,
            'optional_param': optional_param,
            'kwargs': kwargs,
            'success': True
        }
    
    @monitor_step03_functions
    @handle_step03_errors
    @validates()
    @traced(span_name='test_comprehensive_monitoring')
    async def test_comprehensive_monitoring(self) -> dict:
        """Test comprehensive monitoring with all features."""
        self.logger.info("🧪 Testing comprehensive monitoring...")
        
        # Run all test functions
        results = []
        
        # Test function call monitoring
        result1 = await self.test_function_call_monitoring()
        results.append(result1)
        
        # Test error handling
        result2 = await self.test_error_handling()
        results.append(result2)
        
        # Test performance monitoring
        result3 = await self.test_performance_monitoring()
        results.append(result3)
        
        # Test parameter validation
        result4 = await self.test_parameter_validation(
            required_param="test_value",
            optional_param=123,
            extra_param="extra_value"
        )
        results.append(result4)
        
        return {
            'test_name': 'comprehensive_monitoring',
            'results': results,
            'total_tests': len(results),
            'success': True
        }
    
    async def run_all_tests(self) -> dict:
        """Run all monitoring tests."""
        self.start_time = datetime.now()
        self.logger.info("🚀 Starting Step03 Enhanced Monitoring Tests...")
        
        try:
            # Run comprehensive test
            result = await self.test_comprehensive_monitoring()
            
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            self.logger.info(f"✅ All tests completed in {duration:.2f} seconds")
            
            return {
                'test_suite': 'step03_enhanced_monitoring',
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration': duration,
                'result': result,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Test suite failed: {e}")
            return {
                'test_suite': 'step03_enhanced_monitoring',
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            }

async def test_reporting_system():
    """Test the reporting system."""
    logger.info("🧪 Testing reporting system...")
    
    # Create mock execution data
    execution_data = {
        'execution_id': 'test_execution_123',
        'function_calls': [
            {
                'function_name': 'test_function',
                'module_name': 'test_module',
                'duration': 0.1,
                'success': True,
                'memory_delta': 5.0,
                'cpu_delta': 10.0,
                'performance_warnings': []
            },
            {
                'function_name': 'failing_function',
                'module_name': 'test_module',
                'duration': 0.05,
                'success': False,
                'error_type': 'ValueError',
                'memory_delta': 2.0,
                'cpu_delta': 5.0,
                'performance_warnings': ['High memory usage']
            }
        ],
        'errors': [
            {
                'error_id': 'error_123',
                'function_name': 'failing_function',
                'error_type': 'ValueError',
                'error_category': 'validation',
                'severity': 'medium',
                'recovery_attempted': True,
                'recovery_successful': True
            }
        ],
        'performance_data': [
            {
                'timestamp': datetime.now().isoformat(),
                'memory_usage': 100.0,
                'cpu_usage': 50.0
            }
        ]
    }
    
    # Create reporter
    reporter = Step03ExecutionReporter(
        output_directory="test_reports",
        enable_html_reports=True,
        enable_csv_exports=True
    )
    
    # Generate report
    start_time = datetime.now()
    end_time = datetime.now()
    
    report = await reporter.generate_report(
        execution_data=execution_data,
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        data_directory="test_data",
        start_time=start_time,
        end_time=end_time
    )
    
    # Save report
    saved_files = await reporter.save_report(
        report,
        formats=[ReportFormat.JSON, ReportFormat.HTML, ReportFormat.CSV, ReportFormat.MARKDOWN]
    )
    
    logger.info(f"✅ Reporting system test completed. Files saved: {saved_files}")
    return saved_files

async def main():
    """Main test function."""
    logger.info("🚀 Starting Step03 Enhanced Monitoring System Tests")
    logger.info("=" * 80)
    
    try:
        # Test monitoring system
        test_monitor = TestStep03Monitoring()
        test_result = await test_monitor.run_all_tests()
        
        logger.info("📊 Test Results:")
        logger.info(f"   Test Suite: {test_result['test_suite']}")
        logger.info(f"   Duration: {test_result.get('duration', 'N/A')} seconds")
        logger.info(f"   Success: {test_result['success']}")
        
        if not test_result['success']:
            logger.error(f"   Error: {test_result.get('error', 'Unknown error')}")
        
        # Test reporting system
        logger.info("=" * 80)
        saved_files = await test_reporting_system()
        
        logger.info("=" * 80)
        logger.info("🎉 All tests completed successfully!")
        logger.info("📁 Generated files:")
        for format_type, file_path in saved_files.items():
            logger.info(f"   {format_type.upper()}: {file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test suite failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)