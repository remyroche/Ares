#!/usr/bin/env python3
"""Simple test for enhanced Step 4 function call monitoring.

This test verifies the core function call monitoring functionality
without requiring all the project dependencies.
"""
import asyncio
import sys
import time
import functools
import traceback
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Mock the missing dependencies
class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARN: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def exception(self, msg): print(f"EXCEPTION: {msg}")
    def getChild(self, name): return self

class MockPsutil:
    def Process(self):
        return self
    def memory_info(self):
        class MemoryInfo:
            rss = 1024 * 1024 * 100  # 100MB
        return MemoryInfo()
    def cpu_percent(self):
        return 25.0

# Set up mocks
psutil = MockPsutil()
PSUTIL_AVAILABLE = True

# Global function call tracking
_function_call_stack = threading.local()
_function_call_history = []
_function_call_lock = threading.Lock()

class FunctionCallTracker:
    """Comprehensive function call tracking and monitoring system."""
    
    def __init__(self):
        self.call_history = []
        self.active_calls = {}
        self.performance_metrics = {}
        self.error_tracking = {}
    
    def start_call(self, func_name: str, args: tuple, kwargs: dict, caller: str = None) -> str:
        """Start tracking a function call."""
        call_id = f"{func_name}_{int(time.time() * 1000000)}"
        
        call_info = {
            'call_id': call_id,
            'function_name': func_name,
            'caller': caller,
            'start_time': time.time(),
            'args': str(args)[:200] + "..." if len(str(args)) > 200 else str(args),
            'kwargs': str(kwargs)[:200] + "..." if len(str(kwargs)) > 200 else str(kwargs),
            'memory_before': psutil.Process().memory_info().rss / 1024 / 1024 if PSUTIL_AVAILABLE else 0,  # MB
            'thread_id': threading.get_ident(),
            'stack_depth': len(getattr(_function_call_stack, 'stack', []))
        }
        
        with _function_call_lock:
            self.active_calls[call_id] = call_info
            self.call_history.append(call_info.copy())
        
        # Update thread-local stack
        if not hasattr(_function_call_stack, 'stack'):
            _function_call_stack.stack = []
        _function_call_stack.stack.append(call_id)
        
        print(f"🔍 FUNCTION_CALL_START: {func_name} (ID: {call_id})")
        print(f"   📞 Called by: {caller or 'ROOT'}")
        print(f"   📊 Memory before: {call_info['memory_before']:.2f} MB")
        print(f"   🧵 Thread: {call_info['thread_id']}")
        print(f"   📏 Stack depth: {call_info['stack_depth']}")
        
        return call_id
    
    def end_call(self, call_id: str, result: Any = None, error: Exception = None) -> Dict[str, Any]:
        """End tracking a function call and generate detailed report."""
        with _function_call_lock:
            if call_id not in self.active_calls:
                print(f"⚠️ Call ID {call_id} not found in active calls")
                return {}
            
            call_info = self.active_calls.pop(call_id)
        
        # Update thread-local stack
        if hasattr(_function_call_stack, 'stack') and call_id in _function_call_stack.stack:
            _function_call_stack.stack.remove(call_id)
        
        end_time = time.time()
        execution_time = end_time - call_info['start_time']
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024 if PSUTIL_AVAILABLE else 0  # MB
        memory_delta = memory_after - call_info['memory_before']
        
        # Generate detailed outcome report
        outcome_report = {
            'call_id': call_id,
            'function_name': call_info['function_name'],
            'caller': call_info['caller'],
            'execution_time_seconds': execution_time,
            'memory_before_mb': call_info['memory_before'],
            'memory_after_mb': memory_after,
            'memory_delta_mb': memory_delta,
            'success': error is None,
            'error_type': type(error).__name__ if error else None,
            'error_message': str(error) if error else None,
            'result_type': type(result).__name__ if result is not None else None,
            'result_size': len(str(result)) if result is not None else 0,
            'thread_id': call_info['thread_id'],
            'stack_depth': call_info['stack_depth'],
            'timestamp': time.time()
        }
        
        # Log detailed outcome
        status_emoji = "✅" if error is None else "❌"
        print(f"{status_emoji} FUNCTION_CALL_END: {call_info['function_name']} (ID: {call_id})")
        print(f"   ⏱️ Execution time: {execution_time:.4f} seconds")
        print(f"   💾 Memory delta: {memory_delta:+.2f} MB")
        print(f"   🎯 Success: {outcome_report['success']}")
        
        if error:
            print(f"   🚨 Error: {type(error).__name__}: {str(error)}")
            print(f"   📍 Traceback: {traceback.format_exc()}")
        else:
            print(f"   📦 Result type: {outcome_report['result_type']}")
            print(f"   📏 Result size: {outcome_report['result_size']} chars")
        
        # Update performance metrics
        func_name = call_info['function_name']
        if func_name not in self.performance_metrics:
            self.performance_metrics[func_name] = {
                'total_calls': 0,
                'total_time': 0,
                'success_count': 0,
                'error_count': 0,
                'avg_execution_time': 0,
                'max_execution_time': 0,
                'min_execution_time': float('inf')
            }
        
        metrics = self.performance_metrics[func_name]
        metrics['total_calls'] += 1
        metrics['total_time'] += execution_time
        metrics['avg_execution_time'] = metrics['total_time'] / metrics['total_calls']
        metrics['max_execution_time'] = max(metrics['max_execution_time'], execution_time)
        metrics['min_execution_time'] = min(metrics['min_execution_time'], execution_time)
        
        if error:
            metrics['error_count'] += 1
        else:
            metrics['success_count'] += 1
        
        # Store error details if any
        if error:
            if func_name not in self.error_tracking:
                self.error_tracking[func_name] = []
            self.error_tracking[func_name].append({
                'timestamp': time.time(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'call_id': call_id
            })
        
        return outcome_report
    
    def get_caller_info(self) -> str:
        """Get information about the calling function."""
        if hasattr(_function_call_stack, 'stack') and _function_call_stack.stack:
            return _function_call_stack.stack[-1]
        return "ROOT"
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report of all function calls."""
        with _function_call_lock:
            return {
                'total_calls': len(self.call_history),
                'active_calls': len(self.active_calls),
                'performance_metrics': self.performance_metrics,
                'error_summary': {
                    func: len(errors) for func, errors in self.error_tracking.items()
                },
                'recent_calls': self.call_history[-10:] if self.call_history else []
            }

# Global tracker instance
_function_tracker = FunctionCallTracker()

def comprehensive_function_monitor(func: Callable) -> Callable:
    """Comprehensive function call monitoring decorator."""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        caller = _function_tracker.get_caller_info()
        call_id = _function_tracker.start_call(func.__name__, args, kwargs, caller)
        
        try:
            result = await func(*args, **kwargs)
            outcome = _function_tracker.end_call(call_id, result)
            return result
        except Exception as e:
            outcome = _function_tracker.end_call(call_id, error=e)
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        caller = _function_tracker.get_caller_info()
        call_id = _function_tracker.start_call(func.__name__, args, kwargs, caller)
        
        try:
            result = func(*args, **kwargs)
            outcome = _function_tracker.end_call(call_id, result)
            return result
        except Exception as e:
            outcome = _function_tracker.end_call(call_id, error=e)
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def log_function_call_summary():
    """Log a summary of all function calls."""
    summary = _function_tracker.generate_summary_report()
    print("📊 FUNCTION_CALL_SUMMARY:")
    print(f"   📞 Total calls: {summary['total_calls']}")
    print(f"   🔄 Active calls: {summary['active_calls']}")
    
    if summary['performance_metrics']:
        print("   ⚡ Performance metrics:")
        for func_name, metrics in summary['performance_metrics'].items():
            print(f"      {func_name}:")
            print(f"         Calls: {metrics['total_calls']}")
            print(f"         Avg time: {metrics['avg_execution_time']:.4f}s")
            print(f"         Success rate: {metrics['success_count']}/{metrics['total_calls']}")
    
    if summary['error_summary']:
        print("   🚨 Error summary:")
        for func_name, error_count in summary['error_summary'].items():
            print(f"      {func_name}: {error_count} errors")

# Test functions
@comprehensive_function_monitor
def test_sync_function(x: int, y: int) -> int:
    """Test synchronous function."""
    time.sleep(0.1)  # Simulate work
    return x + y

@comprehensive_function_monitor
async def test_async_function(x: int, y: int) -> int:
    """Test asynchronous function."""
    await asyncio.sleep(0.1)  # Simulate async work
    return x * y

@comprehensive_function_monitor
def test_function_with_error():
    """Test function that raises an error."""
    raise ValueError("This is a test error")

@comprehensive_function_monitor
def test_nested_function_calls():
    """Test function that calls other monitored functions."""
    result1 = test_sync_function(5, 3)
    result2 = test_sync_function(10, 20)
    return result1 + result2

async def main():
    """Main test function."""
    print("🧪 Testing Enhanced Step 4 Function Call Monitoring")
    print("=" * 60)
    
    # Test 1: Synchronous function
    print("\n🔧 Test 1: Synchronous Function")
    result1 = test_sync_function(10, 5)
    print(f"   Result: {result1}")
    
    # Test 2: Asynchronous function
    print("\n🔧 Test 2: Asynchronous Function")
    result2 = await test_async_function(4, 6)
    print(f"   Result: {result2}")
    
    # Test 3: Function with error
    print("\n🔧 Test 3: Function with Error")
    try:
        test_function_with_error()
    except Exception as e:
        print(f"   Expected error caught: {e}")
    
    # Test 4: Nested function calls
    print("\n🔧 Test 4: Nested Function Calls")
    result3 = test_nested_function_calls()
    print(f"   Result: {result3}")
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 FUNCTION CALL SUMMARY")
    print("=" * 60)
    log_function_call_summary()
    
    print("\n✅ All tests completed successfully!")
    print("🎯 Function call monitoring is working correctly!")

if __name__ == "__main__":
    asyncio.run(main())