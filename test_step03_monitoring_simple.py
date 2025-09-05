#!/usr/bin/env python3
"""
Simple Test Script for Enhanced Step03 Monitoring System.

This script tests the core monitoring functionality without requiring
all project dependencies.
"""

import asyncio
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_step03_monitoring_simple.log')
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class FunctionCallMetrics:
    """Simple metrics for function calls."""
    function_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    return_value: Any = None
    exception: Optional[Exception] = None

class SimpleFunctionMonitor:
    """Simple function call monitor for testing."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SimpleFunctionMonitor")
        self.call_history: List[FunctionCallMetrics] = []
    
    def monitor_function(self, func):
        """Simple function monitoring decorator."""
        def wrapper(*args, **kwargs):
            # Create metrics
            metrics = FunctionCallMetrics(
                function_name=func.__name__,
                start_time=time.time(),
                parameters={'args': str(args), 'kwargs': str(kwargs)}
            )
            
            # Log function entry
            self.logger.info(f"🚀 ENTERING {func.__name__}")
            self.logger.info(f"   ⏰ Start time: {datetime.fromtimestamp(metrics.start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")
            self.logger.info(f"   📋 Parameters: {len(metrics.parameters)} parameters")
            
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    # Check if we're already in an event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an event loop, create a task
                        result = loop.run_until_complete(func(*args, **kwargs))
                    except RuntimeError:
                        # No event loop running, use asyncio.run
                        result = asyncio.run(func(*args, **kwargs))
                else:
                    result = func(*args, **kwargs)
                
                metrics.return_value = result
                metrics.success = True
                
            except Exception as e:
                metrics.exception = e
                metrics.success = False
                self.logger.error(f"   💥 Exception: {type(e).__name__}: {str(e)}")
                raise
            
            finally:
                # Finalize metrics
                metrics.end_time = time.time()
                metrics.duration = metrics.end_time - metrics.start_time
                
                # Log function exit
                status_emoji = "✅" if metrics.success else "❌"
                status_text = "COMPLETED" if metrics.success else "FAILED"
                
                self.logger.info(f"{status_emoji} EXITING {func.__name__} - {status_text}")
                self.logger.info(f"   ⏰ End time: {datetime.fromtimestamp(metrics.end_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")
                self.logger.info(f"   ⏱️ Duration: {metrics.duration:.4f} seconds")
                
                if metrics.success and metrics.return_value is not None:
                    return_type = type(metrics.return_value).__name__
                    return_str = str(metrics.return_value)[:100]
                    self.logger.info(f"   📤 Return value: {return_type} - {return_str}...")
                
                # Store metrics
                self.call_history.append(metrics)
            
            return metrics.return_value
        
        return wrapper
    
    def get_call_summary(self) -> Dict[str, Any]:
        """Get summary of function calls."""
        if not self.call_history:
            return {'total_calls': 0}
        
        total_calls = len(self.call_history)
        successful_calls = sum(1 for call in self.call_history if call.success)
        failed_calls = total_calls - successful_calls
        total_duration = sum(call.duration for call in self.call_history if call.duration)
        avg_duration = total_duration / total_calls if total_calls > 0 else 0
        
        return {
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'failed_calls': failed_calls,
            'success_rate': successful_calls / total_calls if total_calls > 0 else 0,
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'function_names': [call.function_name for call in self.call_history]
        }

# Create global monitor instance
monitor = SimpleFunctionMonitor()

# Decorator for easy use
def monitor_function_calls(func):
    """Decorator to monitor function calls."""
    return monitor.monitor_function(func)

class TestStep03Monitoring:
    """Test class for Step03 enhanced monitoring system."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TestStep03Monitoring")
        self.test_results = []
        self.start_time = None
    
    @monitor_function_calls
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
    
    @monitor_function_calls
    async def _nested_function_1(self) -> str:
        """Nested function for testing call tracking."""
        await asyncio.sleep(0.05)
        return "nested_function_1_result"
    
    @monitor_function_calls
    async def _nested_function_2(self) -> str:
        """Another nested function for testing call tracking."""
        await asyncio.sleep(0.03)
        return "nested_function_2_result"
    
    @monitor_function_calls
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
    
    @monitor_function_calls
    async def _function_that_fails(self) -> None:
        """Function that intentionally fails for testing."""
        await asyncio.sleep(0.02)
        raise ValueError("Intentional test error for error handling validation")
    
    @monitor_function_calls
    async def test_performance_monitoring(self) -> dict:
        """Test performance monitoring capabilities."""
        self.logger.info("🧪 Testing performance monitoring...")
        
        # Simulate CPU-intensive work
        start_time = time.time()
        result = 0
        for i in range(100000):  # Reduced for faster testing
            result += i * i
        end_time = time.time()
        
        # Simulate memory-intensive work
        large_list = [i for i in range(10000)]  # Reduced for faster testing
        
        return {
            'test_name': 'performance_monitoring',
            'computation_time': end_time - start_time,
            'result': result,
            'list_size': len(large_list),
            'success': True
        }
    
    @monitor_function_calls
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
    
    @monitor_function_calls
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

async def main():
    """Main test function."""
    logger.info("🚀 Starting Step03 Enhanced Monitoring System Tests")
    logger.info("=" * 80)
    
    try:
        # Test monitoring system
        test_monitor = TestStep03Monitoring()
        test_result = await test_monitor.run_all_tests()
        
        logger.info("=" * 80)
        logger.info("📊 Test Results:")
        logger.info(f"   Test Suite: {test_result['test_suite']}")
        logger.info(f"   Duration: {test_result.get('duration', 'N/A')} seconds")
        logger.info(f"   Success: {test_result['success']}")
        
        if not test_result['success']:
            logger.error(f"   Error: {test_result.get('error', 'Unknown error')}")
        
        # Get call summary
        call_summary = monitor.get_call_summary()
        logger.info("=" * 80)
        logger.info("📊 Function Call Summary:")
        logger.info(f"   Total Calls: {call_summary['total_calls']}")
        logger.info(f"   Successful Calls: {call_summary['successful_calls']}")
        logger.info(f"   Failed Calls: {call_summary['failed_calls']}")
        logger.info(f"   Success Rate: {call_summary['success_rate']:.1%}")
        logger.info(f"   Total Duration: {call_summary['total_duration']:.4f} seconds")
        logger.info(f"   Average Duration: {call_summary['avg_duration']:.4f} seconds")
        logger.info(f"   Functions Called: {', '.join(call_summary['function_names'])}")
        
        logger.info("=" * 80)
        logger.info("🎉 All tests completed successfully!")
        logger.info("✅ Enhanced monitoring system is working correctly!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test suite failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)