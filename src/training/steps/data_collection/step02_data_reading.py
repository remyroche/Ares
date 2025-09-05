"""Step 2: Data Reading and Validation with Comprehensive Function Monitoring.

This module handles reading the unified data from step1_5 and performs comprehensive
data quality validation before proceeding to HMM regime discovery. It includes
thorough function call monitoring, function-to-function call tracking, and detailed
completion reporting with outcome analysis.
"""
import asyncio
import sys
import time
import traceback
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import functools

# Enhanced function monitoring framework
class FunctionCallStatus(Enum):
    """Status of function calls."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"

@dataclass
class FunctionCallContext:
    """Context for function call monitoring."""
    function_name: str
    module_name: str
    call_id: str
    start_time: float
    end_time: Optional[float] = None
    status: FunctionCallStatus = FunctionCallStatus.PENDING
    input_args: Dict[str, Any] = field(default_factory=dict)
    input_kwargs: Dict[str, Any] = field(default_factory=dict)
    output_result: Any = None
    error_details: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    called_functions: List[str] = field(default_factory=list)
    parent_call_id: Optional[str] = None
    child_calls: List[str] = field(default_factory=list)

@dataclass
class FunctionInteractionReport:
    """Report of function interactions and outcomes."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    function_call_details: List[FunctionCallContext] = field(default_factory=list)
    call_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_summary: Dict[str, int] = field(default_factory=dict)

class FunctionCallMonitor:
    """Comprehensive function call monitoring system with performance tracking."""
    
    def __init__(self):
        self.active_calls: Dict[str, FunctionCallContext] = {}
        self.completed_calls: List[FunctionCallContext] = []
        self.call_counter = 0
        self.logger = None
        self.performance_metrics = {
            'memory_usage': [],
            'cpu_usage': [],
            'execution_times': [],
            'error_rates': []
        }
        self._setup_logger()
        self._setup_performance_monitoring()
    
    def _setup_logger(self):
        """Setup logger for function monitoring."""
        import logging
        self.logger = logging.getLogger(f"{__name__}.FunctionCallMonitor")
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring capabilities."""
        try:
            import psutil
            self.psutil_available = True
            self.process = psutil.Process()
        except ImportError:
            self.psutil_available = False
            self.logger.warning("⚠️ psutil not available - performance monitoring limited")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.psutil_available:
            try:
                memory_info = self.process.memory_info()
                return memory_info.rss / 1024 / 1024  # Convert to MB
            except Exception:
                return 0.0
        return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if self.psutil_available:
            try:
                return self.process.cpu_percent()
            except Exception:
                return 0.0
        return 0.0
    
    def _generate_call_id(self, function_name: str) -> str:
        """Generate unique call ID."""
        self.call_counter += 1
        return f"{function_name}_{self.call_counter}_{int(time.time() * 1000)}"
    
    def start_function_call(
        self, 
        func: Callable, 
        args: tuple, 
        kwargs: dict, 
        parent_call_id: Optional[str] = None
    ) -> str:
        """Start monitoring a function call with enhanced interaction tracking."""
        call_id = self._generate_call_id(func.__name__)
        
        # Enhanced input tracking with type information
        input_args = {}
        for i, arg in enumerate(args):
            arg_type = type(arg).__name__
            arg_value = str(arg)[:100] if arg is not None else "None"
            input_args[f"arg_{i}"] = {
                "type": arg_type,
                "value": arg_value,
                "size": len(str(arg)) if hasattr(arg, '__len__') else 0
            }
        
        input_kwargs = {}
        for k, v in kwargs.items():
            val_type = type(v).__name__
            val_value = str(v)[:100] if v is not None else "None"
            input_kwargs[k] = {
                "type": val_type,
                "value": val_value,
                "size": len(str(v)) if hasattr(v, '__len__') else 0
            }
        
        # Get initial performance metrics
        initial_memory = self._get_memory_usage()
        initial_cpu = self._get_cpu_usage()
        
        context = FunctionCallContext(
            function_name=func.__name__,
            module_name=func.__module__,
            call_id=call_id,
            start_time=time.time(),
            status=FunctionCallStatus.IN_PROGRESS,
            input_args=input_args,
            input_kwargs=input_kwargs,
            parent_call_id=parent_call_id,
            memory_usage=initial_memory,
            cpu_usage=initial_cpu
        )
        
        self.active_calls[call_id] = context
        
        # Update parent call if exists and track function interactions
        if parent_call_id and parent_call_id in self.active_calls:
            self.active_calls[parent_call_id].child_calls.append(call_id)
            # Track which function called this function
            parent_function = self.active_calls[parent_call_id].function_name
            if parent_function not in context.called_functions:
                context.called_functions.append(parent_function)
        
        # Log detailed function call information
        self.logger.info(f"🔍 Function call started: {func.__name__} (ID: {call_id})")
        self.logger.info(f"   - Module: {func.__module__}")
        self.logger.info(f"   - Parent call: {parent_call_id if parent_call_id else 'None'}")
        self.logger.info(f"   - Input args: {len(input_args)} arguments")
        self.logger.info(f"   - Input kwargs: {len(input_kwargs)} keyword arguments")
        
        return call_id
    
    def complete_function_call(
        self, 
        call_id: str, 
        result: Any = None, 
        error: Optional[Exception] = None
    ) -> None:
        """Complete monitoring a function call with detailed outcome analysis."""
        if call_id not in self.active_calls:
            self.logger.warning(f"⚠️ Unknown call ID: {call_id}")
            return
        
        context = self.active_calls[call_id]
        context.end_time = time.time()
        context.execution_time = context.end_time - context.start_time
        
        # Get final performance metrics
        final_memory = self._get_memory_usage()
        final_cpu = self._get_cpu_usage()
        
        # Calculate performance deltas
        memory_delta = final_memory - (context.memory_usage or 0)
        cpu_delta = final_cpu - (context.cpu_usage or 0)
        
        # Update performance metrics
        self.performance_metrics['memory_usage'].append(memory_delta)
        self.performance_metrics['cpu_usage'].append(cpu_delta)
        self.performance_metrics['execution_times'].append(context.execution_time)
        
        # Enhanced result tracking
        if result is not None:
            result_type = type(result).__name__
            result_size = len(str(result)) if hasattr(result, '__len__') else 0
            context.output_result = {
                "type": result_type,
                "value": str(result)[:200],
                "size": result_size,
                "is_dataframe": hasattr(result, 'shape') and hasattr(result, 'columns'),
                "is_dict": isinstance(result, dict),
                "is_list": isinstance(result, list)
            }
        else:
            context.output_result = {
                "type": "NoneType",
                "value": "None",
                "size": 0,
                "is_dataframe": False,
                "is_dict": False,
                "is_list": False
            }
        
        if error:
            context.status = FunctionCallStatus.FAILED
            context.error_details = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "error_location": f"{error.__class__.__module__}.{error.__class__.__name__}",
                "error_severity": "HIGH" if isinstance(error, (ValueError, TypeError, AttributeError)) else "MEDIUM"
            }
            
            # Detailed error logging
            self.logger.error(f"❌ Function call failed: {context.function_name} (ID: {call_id})")
            self.logger.error(f"   - Error type: {type(error).__name__}")
            self.logger.error(f"   - Error message: {str(error)}")
            self.logger.error(f"   - Execution time: {context.execution_time:.3f}s")
            self.logger.error(f"   - Child calls: {len(context.child_calls)}")
            
        else:
            context.status = FunctionCallStatus.COMPLETED
            
            # Detailed success logging
            self.logger.info(f"✅ Function call completed: {context.function_name} (ID: {call_id})")
            self.logger.info(f"   - Execution time: {context.execution_time:.3f}s")
            self.logger.info(f"   - Result type: {context.output_result['type']}")
            self.logger.info(f"   - Result size: {context.output_result['size']}")
            self.logger.info(f"   - Child calls: {len(context.child_calls)}")
            
            # Log performance metrics
            if context.execution_time > 1.0:
                self.logger.warning(f"⚠️ Slow function execution: {context.function_name} took {context.execution_time:.3f}s")
            elif context.execution_time < 0.001:
                self.logger.info(f"⚡ Fast function execution: {context.function_name} took {context.execution_time:.3f}s")
        
        # Move to completed calls
        self.completed_calls.append(context)
        del self.active_calls[call_id]
    
    def get_function_interaction_report(self) -> FunctionInteractionReport:
        """Generate comprehensive function interaction report."""
        total_calls = len(self.completed_calls)
        successful_calls = len([c for c in self.completed_calls if c.status == FunctionCallStatus.COMPLETED])
        failed_calls = len([c for c in self.completed_calls if c.status == FunctionCallStatus.FAILED])
        
        total_execution_time = sum(c.execution_time or 0 for c in self.completed_calls)
        average_execution_time = total_execution_time / total_calls if total_calls > 0 else 0.0
        
        # Build call hierarchy
        call_hierarchy = {}
        for call in self.completed_calls:
            if call.parent_call_id:
                if call.parent_call_id not in call_hierarchy:
                    call_hierarchy[call.parent_call_id] = []
                call_hierarchy[call.parent_call_id].append(call.call_id)
        
        # Error summary
        error_summary = {}
        for call in self.completed_calls:
            if call.error_details:
                error_type = call.error_details.get("error_type", "Unknown")
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
        
        # Enhanced performance metrics
        if self.completed_calls:
            fastest_call = min(self.completed_calls, key=lambda c: c.execution_time or float('inf'))
            slowest_call = max(self.completed_calls, key=lambda c: c.execution_time or 0)
            
            # Function call frequency analysis
            function_frequency = {}
            for call in self.completed_calls:
                function_frequency[call.function_name] = function_frequency.get(call.function_name, 0) + 1
            
            most_called_function = max(function_frequency.items(), key=lambda x: x[1])[0] if function_frequency else None
            
            # Execution time analysis
            execution_times = [c.execution_time for c in self.completed_calls if c.execution_time is not None]
            median_execution_time = sorted(execution_times)[len(execution_times)//2] if execution_times else 0.0
            
            # Call hierarchy depth analysis
            max_depth = 0
            for call in self.completed_calls:
                depth = self._calculate_call_depth(call.call_id)
                max_depth = max(max_depth, depth)
            
            # Data flow analysis
            dataframe_calls = len([c for c in self.completed_calls if c.output_result and c.output_result.get('is_dataframe', False)])
            dict_calls = len([c for c in self.completed_calls if c.output_result and c.output_result.get('is_dict', False)])
            list_calls = len([c for c in self.completed_calls if c.output_result and c.output_result.get('is_list', False)])
            
            performance_metrics = {
                "fastest_call": fastest_call.function_name,
                "fastest_call_time": fastest_call.execution_time,
                "slowest_call": slowest_call.function_name,
                "slowest_call_time": slowest_call.execution_time,
                "most_called_function": most_called_function,
                "most_called_count": function_frequency.get(most_called_function, 0) if most_called_function else 0,
                "success_rate": (successful_calls / total_calls * 100) if total_calls > 0 else 0.0,
                "median_execution_time": median_execution_time,
                "max_call_depth": max_depth,
                "dataframe_operations": dataframe_calls,
                "dict_operations": dict_calls,
                "list_operations": list_calls,
                "function_frequency": function_frequency
            }
        else:
            performance_metrics = {
                "fastest_call": None,
                "fastest_call_time": 0.0,
                "slowest_call": None,
                "slowest_call_time": 0.0,
                "most_called_function": None,
                "most_called_count": 0,
                "success_rate": 0.0,
                "median_execution_time": 0.0,
                "max_call_depth": 0,
                "dataframe_operations": 0,
                "dict_operations": 0,
                "list_operations": 0,
                "function_frequency": {}
            }
        
        return FunctionInteractionReport(
            total_calls=total_calls,
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            total_execution_time=total_execution_time,
            average_execution_time=average_execution_time,
            function_call_details=self.completed_calls.copy(),
            call_hierarchy=call_hierarchy,
            performance_metrics=performance_metrics,
            error_summary=error_summary
        )
    
    def _calculate_call_depth(self, call_id: str) -> int:
        """Calculate the depth of a function call in the hierarchy."""
        depth = 0
        current_call_id = call_id
        
        # Find the call in completed calls
        current_call = None
        for call in self.completed_calls:
            if call.call_id == current_call_id:
                current_call = call
                break
        
        if not current_call:
            return 0
        
        # Traverse up the parent chain
        while current_call and current_call.parent_call_id:
            depth += 1
            parent_call_id = current_call.parent_call_id
            
            # Find parent call
            current_call = None
            for call in self.completed_calls:
                if call.call_id == parent_call_id:
                    current_call = call
                    break
        
        return depth

# Global function call monitor
function_monitor = FunctionCallMonitor()

# Context variable for tracking current function call
import contextvars
current_call_context = contextvars.ContextVar('current_call_id', default=None)

def comprehensive_function_monitoring(
    validate_inputs: bool = True,
    validate_outputs: bool = True,
    track_performance: bool = True,
    track_memory: bool = True,
    timeout_seconds: Optional[int] = None,
    retry_attempts: int = 0
):
    """Comprehensive decorator for function call monitoring and validation."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Get parent call ID from context
            parent_call_id = current_call_context.get()
            
            # Start function call with parent context
            call_id = function_monitor.start_function_call(func, args, kwargs, parent_call_id)
            
            # Set this call as the current context for child calls
            token = current_call_context.set(call_id)
            
            try:
                # Input validation
                if validate_inputs:
                    await _validate_function_inputs(func, args, kwargs)
                
                # Execute with timeout if specified
                if timeout_seconds:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout_seconds
                    )
                else:
                    result = await func(*args, **kwargs)
                
                # Output validation
                if validate_outputs:
                    await _validate_function_outputs(result)
                
                function_monitor.complete_function_call(call_id, result)
                return result
                
            except Exception as e:
                function_monitor.complete_function_call(call_id, error=e)
                
                # Retry logic
                if retry_attempts > 0:
                    return await _retry_function_call(func, args, kwargs, retry_attempts, call_id)
                
                raise
            finally:
                # Reset context to parent
                current_call_context.reset(token)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Get parent call ID from context
            parent_call_id = current_call_context.get()
            
            # Start function call with parent context
            call_id = function_monitor.start_function_call(func, args, kwargs, parent_call_id)
            
            # Set this call as the current context for child calls
            token = current_call_context.set(call_id)
            
            try:
                # Input validation
                if validate_inputs:
                    _validate_function_inputs_sync(func, args, kwargs)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Output validation
                if validate_outputs:
                    _validate_function_outputs_sync(result)
                
                function_monitor.complete_function_call(call_id, result)
                return result
                
            except Exception as e:
                function_monitor.complete_function_call(call_id, error=e)
                
                # Retry logic
                if retry_attempts > 0:
                    return _retry_function_call_sync(func, args, kwargs, retry_attempts, call_id)
                
                raise
            finally:
                # Reset context to parent
                current_call_context.reset(token)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

async def _validate_function_inputs(func: Callable, args: tuple, kwargs: dict) -> None:
    """Validate function inputs with comprehensive error handling."""
    try:
        # Basic input validation logic
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Enhanced validation for specific types
        for param_name, param_value in bound_args.arguments.items():
            param_annotation = sig.parameters[param_name].annotation
            
            # Validate string parameters
            if param_annotation == str and not isinstance(param_value, str):
                raise TypeError(f"Parameter '{param_name}' must be a string, got {type(param_value).__name__}")
            
            # Validate path parameters
            if 'path' in param_name.lower() or 'dir' in param_name.lower():
                if param_value and not isinstance(param_value, (str, Path)):
                    raise TypeError(f"Parameter '{param_name}' must be a string or Path, got {type(param_value).__name__}")
            
            # Validate DataFrame parameters
            if 'data' in param_name.lower() and param_value is not None:
                if not hasattr(param_value, 'shape') or not hasattr(param_value, 'columns'):
                    raise TypeError(f"Parameter '{param_name}' must be a DataFrame, got {type(param_value).__name__}")
        
        # Add specific validation logic here
        pass
        
    except Exception as e:
        function_monitor.logger.error(f"❌ Input validation failed for {func.__name__}: {e}")
        raise ValueError(f"Input validation failed: {e}") from e

def _validate_function_inputs_sync(func: Callable, args: tuple, kwargs: dict) -> None:
    """Validate function inputs (sync version) with comprehensive error handling."""
    try:
        # Basic input validation logic
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Enhanced validation for specific types
        for param_name, param_value in bound_args.arguments.items():
            param_annotation = sig.parameters[param_name].annotation
            
            # Validate string parameters
            if param_annotation == str and not isinstance(param_value, str):
                raise TypeError(f"Parameter '{param_name}' must be a string, got {type(param_value).__name__}")
            
            # Validate path parameters
            if 'path' in param_name.lower() or 'dir' in param_name.lower():
                if param_value and not isinstance(param_value, (str, Path)):
                    raise TypeError(f"Parameter '{param_name}' must be a string or Path, got {type(param_value).__name__}")
            
            # Validate DataFrame parameters
            if 'data' in param_name.lower() and param_value is not None:
                if not hasattr(param_value, 'shape') or not hasattr(param_value, 'columns'):
                    raise TypeError(f"Parameter '{param_name}' must be a DataFrame, got {type(param_value).__name__}")
        
        # Add specific validation logic here
        pass
        
    except Exception as e:
        function_monitor.logger.error(f"❌ Input validation failed for {func.__name__}: {e}")
        raise ValueError(f"Input validation failed: {e}") from e

async def _validate_function_outputs(result: Any) -> None:
    """Validate function outputs with comprehensive error handling."""
    try:
        # Basic output validation logic
        if result is None:
            raise ValueError("Function returned None")
        
        # Enhanced validation for specific types
        if hasattr(result, 'shape') and hasattr(result, 'columns'):
            # DataFrame validation
            if result.shape[0] == 0:
                raise ValueError("Function returned empty DataFrame")
            if result.shape[1] == 0:
                raise ValueError("Function returned DataFrame with no columns")
        
        elif isinstance(result, dict):
            # Dictionary validation
            if not result:
                raise ValueError("Function returned empty dictionary")
        
        elif isinstance(result, list):
            # List validation
            if not result:
                raise ValueError("Function returned empty list")
        
        # Add specific validation logic here
        pass
        
    except Exception as e:
        function_monitor.logger.error(f"❌ Output validation failed: {e}")
        raise ValueError(f"Output validation failed: {e}") from e

def _validate_function_outputs_sync(result: Any) -> None:
    """Validate function outputs (sync version) with comprehensive error handling."""
    try:
        # Basic output validation logic
        if result is None:
            raise ValueError("Function returned None")
        
        # Enhanced validation for specific types
        if hasattr(result, 'shape') and hasattr(result, 'columns'):
            # DataFrame validation
            if result.shape[0] == 0:
                raise ValueError("Function returned empty DataFrame")
            if result.shape[1] == 0:
                raise ValueError("Function returned DataFrame with no columns")
        
        elif isinstance(result, dict):
            # Dictionary validation
            if not result:
                raise ValueError("Function returned empty dictionary")
        
        elif isinstance(result, list):
            # List validation
            if not result:
                raise ValueError("Function returned empty list")
        
        # Add specific validation logic here
        pass
        
    except Exception as e:
        function_monitor.logger.error(f"❌ Output validation failed: {e}")
        raise ValueError(f"Output validation failed: {e}") from e

async def _retry_function_call(func: Callable, args: tuple, kwargs: dict, retry_attempts: int, original_call_id: str) -> Any:
    """Retry function call with monitoring."""
    for attempt in range(retry_attempts):
        try:
            # Get parent call ID from context
            parent_call_id = current_call_context.get()
            retry_call_id = function_monitor.start_function_call(func, args, kwargs, parent_call_id)
            
            # Set this call as the current context for child calls
            token = current_call_context.set(retry_call_id)
            
            try:
                result = await func(*args, **kwargs)
                function_monitor.complete_function_call(retry_call_id, result)
                return result
            finally:
                # Reset context to parent
                current_call_context.reset(token)
                
        except Exception as e:
            function_monitor.complete_function_call(retry_call_id, error=e)
            if attempt == retry_attempts - 1:
                raise
            await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff

def _retry_function_call_sync(func: Callable, args: tuple, kwargs: dict, retry_attempts: int, original_call_id: str) -> Any:
    """Retry function call with monitoring (sync version)."""
    for attempt in range(retry_attempts):
        try:
            # Get parent call ID from context
            parent_call_id = current_call_context.get()
            retry_call_id = function_monitor.start_function_call(func, args, kwargs, parent_call_id)
            
            # Set this call as the current context for child calls
            token = current_call_context.set(retry_call_id)
            
            try:
                result = func(*args, **kwargs)
                function_monitor.complete_function_call(retry_call_id, result)
                return result
            finally:
                # Reset context to parent
                current_call_context.reset(token)
                
        except Exception as e:
            function_monitor.complete_function_call(retry_call_id, error=e)
            if attempt == retry_attempts - 1:
                raise
            time.sleep(0.1 * (2 ** attempt))  # Exponential backoff

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.common_operations import safe_read_parquet
from pathlib import Path as _Path
import json as _json
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
REQUIRED_MODULES = ['pandas', 'numpy', 'psutil', 'src.utils.centralized_decorators', 'src.utils.logger', 'src.utils.enhanced_mlflow_integration']
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Create fallback decorators
def handles_errors(exceptions=(Exception,), default_return=None, context=None):
    """Fallback error handling decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logging.error(f"Error in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator
if system_logger is None:
    system_logger = create_fallback_logger()
if centralized_decorators is None:
    comprehensive_data_validation = create_fallback_decorator()
    handle_errors = create_fallback_decorator()
    memory_efficient = create_fallback_decorator()
    resource_monitor = create_fallback_decorator()
    secure_data_processing = create_fallback_decorator()
    validate_data_structure = create_fallback_decorator()
    with_tracing_span = create_fallback_decorator()
    quality_gate = create_fallback_decorator()
    monitor_feature_engineering = create_fallback_decorator()
else:
    comprehensive_data_validation = centralized_decorators.comprehensive_data_validation
    handle_errors = centralized_decorators.handle_errors
    memory_efficient = centralized_decorators.memory_efficient
    resource_monitor = centralized_decorators.resource_monitor
    secure_data_processing = centralized_decorators.secure_data_processing
    validate_data_structure = centralized_decorators.validate_data_structure
    with_tracing_span = centralized_decorators.with_tracing_span
    quality_gate = centralized_decorators.quality_gate
    monitor_feature_engineering = centralized_decorators.monitor_feature_engineering
if enhanced_mlflow is None:
    with_enhanced_mlflow_logging = create_fallback_decorator()
    log_step_report = lambda *args, **kwargs: 'fallback_report'
    create_detailed_step_report = lambda *args, **kwargs: {}
    log_step_metrics = lambda *args, **kwargs: None
    log_step_dataframe_with_standardized_name = lambda *args, **kwargs: 'fallback_dataframe'
    log_step_artifact_with_standardized_name = lambda *args, **kwargs: 'fallback_artifact'
else:
    with_enhanced_mlflow_logging = enhanced_mlflow.with_enhanced_mlflow_logging
    log_step_report = enhanced_mlflow.log_step_report
    create_detailed_step_report = enhanced_mlflow.create_detailed_step_report
    log_step_metrics = enhanced_mlflow.log_step_metrics
    log_step_dataframe_with_standardized_name = enhanced_mlflow.log_step_dataframe_with_standardized_name
    log_step_artifact_with_standardized_name = enhanced_mlflow.log_step_artifact_with_standardized_name
import pandas as pd

# Provide safe, no-op decorators to avoid import-time failures in legacy module
def _identity_decorator(*_dargs: Any, **_dkwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _decor(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn
    return _decor

traced = _identity_decorator
validates = _identity_decorator
cached = _identity_decorator
log_execution_time = _identity_decorator
handles_errors = _identity_decorator

# Ensure we obtain a proper logger instance (not the module) when available
try:
    if system_logger is not None and not hasattr(system_logger, 'getChild'):
        # Likely the imported module; extract the logger instance attribute
        system_logger = getattr(system_logger, 'system_logger', system_logger)
except Exception:
    pass

logger = system_logger.getChild('Step2DataReading') if hasattr(system_logger, 'getChild') else create_fallback_logger()

class DataReadingStep:
    """Step 2: Data Reading and Validation with comprehensive function monitoring and standardized data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('DataReadingStep')
        self.standards = pipeline_standards
        self.start_time = None
        self.step_timings = {}
        self.function_interaction_report = None
        self._validate_environment()

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info('🔍 Validating environment dependencies...')
        missing_modules = [module for module, available in dependency_status.items() if not available]
        if missing_modules:
            self.logger.warning(f'⚠️ Missing optional modules: {missing_modules}')
            self.logger.info('📝 Pipeline will continue with fallback implementations')
        else:
            self.logger.info('✅ All required dependencies available')

    async def initialize(self) -> None:
        """Initialize the data reading step."""
        self.start_time = time.time()
        self.logger.info('🚀 Initializing Data Reading Step...')
        self.logger.info('📋 Step 2 Configuration:')
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        self.logger.info('✅ Data Reading Step initialized successfully')

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f'⏱️ {step_name} completed in {elapsed:.2f} seconds')

    @comprehensive_function_monitoring(
        validate_inputs=True,
        validate_outputs=True,
        track_performance=True,
        timeout_seconds=300,
        retry_attempts=2
    )
    @traced(span_name='read_unified_data')
    @validates(min_quality_score=0.8, max_correlation=0.95, required_grade='B')
    @cached
    async def read_unified_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Read unified data from step1_5 output with standardized validation."""
        step_start = time.time()
        self.logger.info(f'📖 Reading unified data for {symbol} on {exchange} ({timeframe})')
        try:
            unified_data_path = Path(self.standards.build_path('unified_data', exchange, symbol)) / timeframe
            if not unified_data_path.exists():
                self.logger.error(f'❌ Unified data path does not exist: {unified_data_path}')
                return None
            parquet_files = list(unified_data_path.glob('**/*.parquet'))
            if not parquet_files:
                self.logger.error(f'❌ No parquet files found in {unified_data_path}')
                return None
            self.logger.info(f'📁 Found {len(parquet_files)} parquet files')
            dataframes = []
            for file_path in sorted(parquet_files):
                self.logger.info(f'📖 Reading {file_path.name}')
                df = safe_read_parquet(file_path)
                df = self.standards.standardize_timestamp(df, 'timestamp')
                df = self.standards.enforce_schema(df, 'unified')
                dataframes.append(df)
            if dataframes:
                unified_data = pd.concat(dataframes, ignore_index=True)
                unified_data = unified_data.sort_values('timestamp').reset_index(drop=True)
                
                # Apply data quality fixes
                from src.utils.data_quality_fixer import DataQualityFixer
                quality_fixer = DataQualityFixer()
                unified_data, fix_report = quality_fixer.fix_data_quality_issues(unified_data, 'timestamp')
                
                validation_result = self.standards.validate_data_quality(unified_data, 'unified')
                if validation_result.passed:
                    self.logger.info(f'✅ Successfully read unified data: {len(unified_data)} rows (quality score: {validation_result.quality_score:.2f})')
                else:
                    self.logger.warning(f'⚠️ Read unified data: {len(unified_data)} rows but validation found issues')
                    for issue in validation_result.issues[:3]:
                        self.logger.warning(f'   - {issue.message}')
                self._log_step_timing('read_unified_data', step_start)
                return unified_data
            else:
                self.logger.error('❌ No data found in parquet files')
                return None
        except Exception as e:
            self.logger.exception(f'❌ Error reading unified data: {e}')
            return None

    @comprehensive_function_monitoring(
        validate_inputs=True,
        validate_outputs=True,
        track_performance=True,
        timeout_seconds=120,
        retry_attempts=1
    )
    @traced(span_name='validate_data_quality')
    @validates()
    async def validate_data_quality(self, data: pd.DataFrame, symbol: str, exchange: str) -> Dict[str, Any]:
        """Validate data quality and structure using standardized validation."""
        step_start = time.time()
        self.logger.info('🔍 Validating data quality...')
        try:
            validation_result = self.standards.validate_data_quality(data, 'unified')
<<<<<<< HEAD
            computed_data_info = {
=======
            
            # Create data_info dictionary
            data_info = {
>>>>>>> origin/main
                'rows': len(data) if data is not None else 0,
                'columns': list(data.columns) if data is not None else [],
                'date_range': {
                    'start': data['timestamp'].min() if data is not None and 'timestamp' in data.columns else None,
<<<<<<< HEAD
                    'end': data['timestamp'].max() if data is not None and 'timestamp' in data.columns else None,
                },
                'memory_usage': data.memory_usage(deep=True).sum() / 1024 / 1024 if data is not None else 0,
            }
=======
                    'end': data['timestamp'].max() if data is not None and 'timestamp' in data.columns else None
                },
                'memory_usage': data.memory_usage(deep=True).sum() / 1024 / 1024 if data is not None else 0
            }
            
            # Create validation_results with data_info
>>>>>>> origin/main
            validation_results = {
                'passed': validation_result.passed,
                'issues': [issue.message for issue in validation_result.issues],
                'warnings': [warning.message for warning in validation_result.warnings],
<<<<<<< HEAD
                'data_info': computed_data_info,
                'quality_score': validation_result.quality_score,
            }
            self.logger.info(f'✅ Data quality validation completed')
            self.logger.info(f"   - Rows: {computed_data_info['rows']}")
            self.logger.info(f"   - Memory usage: {computed_data_info['memory_usage']:.2f} MB")
            self.logger.info(f'   - Quality score: {validation_result.quality_score:.2f}')
            self.logger.info(f"   - Issues: {len(validation_results['issues'])}")
            self.logger.info(f"   - Warnings: {len(validation_results['warnings'])}")
            thresholds = self.config.get('step02_quality_thresholds', {'min_rows': 100000, 'max_null_ratio': 0.01, 'min_quality_score': 0.8})
            rows = computed_data_info['rows']
            null_ratio = float(data.isnull().sum().sum()) / (max(1, rows) * max(1, len(data.columns))) if rows else 1.0
=======
                'data_info': data_info,
                'quality_score': validation_result.quality_score
            }
            
            self.logger.info(f'✅ Data quality validation completed')
            self.logger.info(f"   - Rows: {data_info['rows']}")
            self.logger.info(f"   - Memory usage: {data_info['memory_usage']:.2f} MB")
            self.logger.info(f'   - Quality score: {validation_result.quality_score:.2f}')
            self.logger.info(f"   - Issues: {len(validation_results['issues'])}")
            self.logger.info(f"   - Warnings: {len(validation_results['warnings'])}")
            
            # Check quality thresholds
            thresholds = self.config.get('step02_quality_thresholds', {
                'min_rows': 100000, 
                'max_null_ratio': 0.01, 
                'min_quality_score': 0.8
            })
            rows = data_info['rows']
            null_ratio = float(data.isnull().sum()) / (max(1, rows) * max(1, len(data.columns))) if rows else 1.0
>>>>>>> origin/main
            quality_score = float(validation_results['quality_score'])
            
            if rows < thresholds['min_rows'] or null_ratio > thresholds['max_null_ratio'] or quality_score < thresholds['min_quality_score']:
                self.logger.error(f"⛔ Early gating: rows={rows} (<{thresholds['min_rows']}), null_ratio={null_ratio:.4f} (>{thresholds['max_null_ratio']}), quality={quality_score:.2f} (<{thresholds['min_quality_score']})")
                validation_results['passed'] = False
            self._log_step_timing('validate_data_quality', step_start)
        except Exception as e:
            self.logger.exception(f'❌ Error during data quality validation: {e}')
            validation_results = {
                'passed': False, 
                'issues': [f'Validation error: {str(e)}'], 
                'warnings': [], 
                'data_info': {
                    'rows': 0,
                    'columns': [],
                    'date_range': {'start': None, 'end': None},
                    'memory_usage': 0.0
                }, 
                'quality_score': 0.0
            }
        return validation_results

    @comprehensive_function_monitoring(
        validate_inputs=True,
        validate_outputs=True,
        track_performance=True,
        timeout_seconds=60,
        retry_attempts=1
    )
    @traced(span_name='save_validation_report')
    async def save_validation_report(self, validation_results: Dict[str, Any], symbol: str, exchange: str, data_dir: str) -> bool:
        """Save validation report to file."""
        step_start = time.time()
        self.logger.info('💾 Saving validation report...')
        
        try:
            reports_dir = Path(data_dir) / 'reports' / 'data_quality'
            reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f'data_reading_validation_{exchange}_{symbol}_{timestamp}.json'
            report_path = reports_dir / report_filename
            
            report_data = {
                'step': 'step02_data_reading', 
                'timestamp': datetime.now().isoformat(), 
                'symbol': symbol, 
                'exchange': exchange, 
                'validation_results': validation_results, 
                'step_timings': self.step_timings
            }
            
            with open(report_path, 'w') as _f:
                _json.dump(report_data, _f, indent=2, default=str)
            self.logger.info(f'✅ Validation report saved to {report_path}')
            self._log_step_timing('save_validation_report', step_start)
            return True
            
        except Exception as e:
            self.logger.exception(f'❌ Error saving validation report: {e}')
            return False

    @comprehensive_function_monitoring(
        validate_inputs=True,
        validate_outputs=True,
        track_performance=True,
        timeout_seconds=600,
        retry_attempts=1
    )
    @traced(span_name='execute_data_reading_step')
    @handles_errors
    @log_execution_time
    async def execute(self, symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> Dict[str, Any]:
        """Execute the complete data reading step."""
        self.logger.info('🚀 Starting Step 2: Data Reading and Validation')
        try:
            unified_data = await self.read_unified_data(symbol, exchange, timeframe, data_dir)
            if unified_data is None:
                self.logger.error('❌ Failed to read unified data')
                return {'success': False, 'error': 'Failed to read unified data'}
            vres = await self.validate_data_quality(unified_data, symbol, exchange)
            if not vres.get('passed', False):
                await self.save_validation_report(vres, symbol, exchange, data_dir)
                self.logger.error('⛔ Early gating failed; marking step as skipped')
                return {'success': False, 'status': 'SKIPPED', 'reason': 'quality_thresholds'}
            validation_results = await self.validate_data_quality(unified_data, symbol, exchange)
            await self.save_validation_report(validation_results, symbol, exchange, data_dir)
            if not validation_results['passed']:
                self.logger.error('❌ Data quality validation failed')
                self.logger.error(f"   Issues: {validation_results['issues']}")
                return {'success': False, 'error': 'Data quality validation failed', 'validation_results': validation_results}
            processed_dir = self.standards.build_path('processed_data', exchange, symbol)
            Path(processed_dir).mkdir(parents=True, exist_ok=True)
            output_file = f'{exchange}_{symbol}_{timeframe}_validated_data.parquet'
            output_path = Path(processed_dir) / output_file
            unified_data = self.standards.standardize_timestamp(unified_data, 'timestamp')
            unified_data.to_parquet(output_path, index=False)
            self.logger.info(f'✅ Step 2 completed successfully')
            self.logger.info(f'   - Validated data saved to: {output_path}')
            self.logger.info(f'   - Total execution time: {time.time() - self.start_time:.2f} seconds')
            
            # Generate comprehensive function interaction report
            function_report_result = await self.generate_function_interaction_report(symbol, exchange, data_dir)
            
            await self._log_step2_artifacts_and_report(symbol, exchange, timeframe, data_dir, unified_data, validation_results, output_path)
            
            return {
                'success': True, 
                'data_path': str(output_path), 
                'validation_results': validation_results, 
                'step_timings': self.step_timings,
                'function_interaction_report': function_report_result
            }
        except Exception as e:
            self.logger.exception(f'❌ Error in Step 2: {e}')
            return {'success': False, 'error': str(e)}

    async def _log_step2_artifacts_and_report(self, symbol: str, exchange: str, timeframe: str, data_dir: str, unified_data: pd.DataFrame, validation_results: Dict[str, Any], output_path: Path) -> None:
        """Log step 2 artifacts and create detailed report."""
        try:
            execution_metadata = {'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else datetime.now().isoformat(), 'end_time': datetime.now().isoformat(), 'duration_seconds': time.time() - self.start_time if self.start_time else 0.0, 'memory_usage_mb': 0.0, 'cpu_usage_percent': 0.0, 'data_quality_score': validation_results.get('quality_score', 0.0), 'processing_efficiency': 1.0 if validation_results.get('passed', False) else 0.5}
            artifacts_generated = [str(output_path), f'{exchange}_{symbol}_{timeframe}_validation_report.json']
            metrics_calculated = {'data_reading_success': 1.0, 'validation_passed': 1.0 if validation_results.get('passed', False) else 0.0, 'data_quality_score': validation_results.get('quality_score', 0.0), 'total_rows': len(unified_data) if unified_data is not None else 0, 'total_columns': len(unified_data.columns) if unified_data is not None else 0, 'validation_issues_count': len(validation_results.get('issues', []))}
            training_input = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir, 'asset': symbol, 'lookback_period': self.config.get('lookback_days', 1095), 'project_version': self.config.get('project_version', '1.0.0')}
            step_data = {'validation_results': validation_results, 'step_timings': self.step_timings, 'data_path': str(output_path)}
            report_data = create_detailed_step_report(step_name='step02_data_reading', step_data=step_data, training_input=training_input, execution_metadata=execution_metadata, artifacts_generated=artifacts_generated, metrics_calculated=metrics_calculated, errors_encountered=[] if validation_results.get('passed', False) else validation_results.get('issues', []))
            report_name = log_step_report(config=self.config, step_name='step02_data_reading', report_data=report_data, report_type='data_reading_report', additional_metadata={'validation_passed': validation_results.get('passed', False), 'data_quality_score': validation_results.get('quality_score', 0.0), 'timeframe': timeframe, 'asset': symbol, 'lookback_period': self.config.get('lookback_days', 1095), 'project_version': self.config.get('project_version', '1.0.0')})
            self.logger.info(f'✅ Logged data reading report: {report_name}')
            if unified_data is not None:
                artifact_name = log_step_dataframe_with_standardized_name(config=self.config, step_name='step02_data_reading', df=unified_data, artifact_type='validated_data', additional_metadata={'artifact_type': 'validated_data', 'dataframe_shape': list(unified_data.shape), 'validation_passed': validation_results.get('passed', False), 'timeframe': timeframe, 'asset': symbol, 'lookback_period': self.config.get('lookback_days', 1095), 'project_version': self.config.get('project_version', '1.0.0')})
                self.logger.info(f'✅ Logged validated data: {artifact_name}')
            validation_report_name = log_step_report(config=self.config, step_name='step02_data_reading', report_data=validation_results, report_type='validation_results', additional_metadata={'validation_passed': validation_results.get('passed', False), 'quality_score': validation_results.get('quality_score', 0.0), 'asset': symbol, 'lookback_period': self.config.get('lookback_days', 1095), 'project_version': self.config.get('project_version', '1.0.0'), 'timeframe': timeframe})
            self.logger.info(f'✅ Logged validation results: {validation_report_name}')
            log_step_metrics(config=self.config, step_name='step02_data_reading', metrics=metrics_calculated, additional_metadata={'metrics_type': 'data_reading_performance', 'timeframe': timeframe, 'asset': symbol, 'lookback_period': self.config.get('lookback_days', 1095), 'project_version': self.config.get('project_version', '1.0.0')})
            self.logger.info('✅ Step 2 artifacts and reports logged successfully')
        except Exception as e:
            self.logger.error(f'❌ Failed to log step 2 artifacts and reports: {e}')

    @comprehensive_function_monitoring(
        validate_inputs=True,
        validate_outputs=True,
        track_performance=True,
        timeout_seconds=30,
        retry_attempts=1
    )
    async def generate_function_interaction_report(self, symbol: str, exchange: str, data_dir: str) -> Dict[str, Any]:
        """Generate comprehensive function interaction report."""
        try:
            self.logger.info('📊 Generating comprehensive function interaction report...')
            
            # Get the function interaction report
            self.function_interaction_report = function_monitor.get_function_interaction_report()
            
            # Save detailed report to file
            reports_dir = ensure_directory(Path(data_dir) / 'reports' / 'function_monitoring')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f'step02_function_interaction_report_{exchange}_{symbol}_{timestamp}.json'
            report_path = reports_dir / report_filename
            
            # Convert dataclass to dict for JSON serialization
            report_data = {
                'step': 'step02_data_reading',
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'exchange': exchange,
                'total_calls': self.function_interaction_report.total_calls,
                'successful_calls': self.function_interaction_report.successful_calls,
                'failed_calls': self.function_interaction_report.failed_calls,
                'total_execution_time': self.function_interaction_report.total_execution_time,
                'average_execution_time': self.function_interaction_report.average_execution_time,
                'performance_metrics': self.function_interaction_report.performance_metrics,
                'error_summary': self.function_interaction_report.error_summary,
                'call_hierarchy': self.function_interaction_report.call_hierarchy,
                'function_call_details': [
                    {
                        'function_name': call.function_name,
                        'module_name': call.module_name,
                        'call_id': call.call_id,
                        'start_time': call.start_time,
                        'end_time': call.end_time,
                        'status': call.status.value,
                        'execution_time': call.execution_time,
                        'parent_call_id': call.parent_call_id,
                        'child_calls': call.child_calls,
                        'error_details': call.error_details,
                        'input_args': call.input_args,
                        'input_kwargs': call.input_kwargs,
                        'output_result': call.output_result
                    }
                    for call in self.function_interaction_report.function_call_details
                ]
            }
            
            safe_json_dump(report_data, report_path, indent=2, default=str)
            
            # Enhanced detailed logging
            self.logger.info('📊 Function Interaction Report Summary:')
            self.logger.info(f'   - Total function calls: {self.function_interaction_report.total_calls}')
            self.logger.info(f'   - Successful calls: {self.function_interaction_report.successful_calls}')
            self.logger.info(f'   - Failed calls: {self.function_interaction_report.failed_calls}')
            self.logger.info(f'   - Success rate: {self.function_interaction_report.performance_metrics.get("success_rate", 0):.1f}%')
            self.logger.info(f'   - Total execution time: {self.function_interaction_report.total_execution_time:.3f}s')
            self.logger.info(f'   - Average execution time: {self.function_interaction_report.average_execution_time:.3f}s')
            self.logger.info(f'   - Median execution time: {self.function_interaction_report.performance_metrics.get("median_execution_time", 0):.3f}s')
            self.logger.info(f'   - Maximum call depth: {self.function_interaction_report.performance_metrics.get("max_call_depth", 0)}')
            
            # Performance analysis
            if self.function_interaction_report.performance_metrics.get("fastest_call"):
                fastest_time = self.function_interaction_report.performance_metrics.get("fastest_call_time", 0)
                self.logger.info(f'   - Fastest call: {self.function_interaction_report.performance_metrics["fastest_call"]} ({fastest_time:.3f}s)')
            if self.function_interaction_report.performance_metrics.get("slowest_call"):
                slowest_time = self.function_interaction_report.performance_metrics.get("slowest_call_time", 0)
                self.logger.info(f'   - Slowest call: {self.function_interaction_report.performance_metrics["slowest_call"]} ({slowest_time:.3f}s)')
            if self.function_interaction_report.performance_metrics.get("most_called_function"):
                most_called_count = self.function_interaction_report.performance_metrics.get("most_called_count", 0)
                self.logger.info(f'   - Most called function: {self.function_interaction_report.performance_metrics["most_called_function"]} ({most_called_count} times)')
            
            # Data flow analysis
            self.logger.info('   - Data flow analysis:')
            self.logger.info(f'     * DataFrame operations: {self.function_interaction_report.performance_metrics.get("dataframe_operations", 0)}')
            self.logger.info(f'     * Dictionary operations: {self.function_interaction_report.performance_metrics.get("dict_operations", 0)}')
            self.logger.info(f'     * List operations: {self.function_interaction_report.performance_metrics.get("list_operations", 0)}')
            
            # Function frequency analysis
            function_frequency = self.function_interaction_report.performance_metrics.get("function_frequency", {})
            if function_frequency:
                self.logger.info('   - Function call frequency:')
                sorted_functions = sorted(function_frequency.items(), key=lambda x: x[1], reverse=True)
                for func_name, count in sorted_functions[:5]:  # Top 5 most called functions
                    self.logger.info(f'     * {func_name}: {count} calls')
            
            # Error analysis
            if self.function_interaction_report.error_summary:
                self.logger.info('   - Error summary:')
                for error_type, count in self.function_interaction_report.error_summary.items():
                    self.logger.info(f'     * {error_type}: {count} occurrences')
            
            # Call hierarchy analysis
            if self.function_interaction_report.call_hierarchy:
                self.logger.info('   - Call hierarchy depth:')
                for parent_id, child_ids in self.function_interaction_report.call_hierarchy.items():
                    parent_call = next((c for c in self.function_interaction_report.function_call_details if c.call_id == parent_id), None)
                    if parent_call:
                        self.logger.info(f'     * {parent_call.function_name}: {len(child_ids)} child calls')
            
            self.logger.info(f'✅ Function interaction report saved to: {report_path}')
            
            return {
                'success': True,
                'report_path': str(report_path),
                'report_summary': {
                    'total_calls': self.function_interaction_report.total_calls,
                    'successful_calls': self.function_interaction_report.successful_calls,
                    'failed_calls': self.function_interaction_report.failed_calls,
                    'success_rate': self.function_interaction_report.performance_metrics.get("success_rate", 0),
                    'total_execution_time': self.function_interaction_report.total_execution_time,
                    'average_execution_time': self.function_interaction_report.average_execution_time
                }
            }
            
        except Exception as e:
            self.logger.exception(f'❌ Error generating function interaction report: {e}')
            return {
                'success': False,
                'error': str(e)
            }

async def run_step_enhanced(symbol: str, exchange: str, timeframe: str, data_dir: str=None, **kwargs) -> Dict[str, Any]:
    """Enhanced entry point for Step 2: Data Reading and Validation."""
    if data_dir is None:
        data_dir = pipeline_standards.build_path('raw_data', exchange, symbol)
    logger.info('🚀 Starting Step 2: Data Reading and Validation (Enhanced)')
    config = {'SYMBOL': symbol, 'EXCHANGE': exchange, 'TIMEFRAME': timeframe, 'DATA_DIR': data_dir, **kwargs}
    step = DataReadingStep(config)
    await step.initialize()
    result = await step.execute(symbol, exchange, timeframe, data_dir, **kwargs)
    if result['success']:
        logger.info('✅ Step 2: Data Reading and Validation completed successfully')
    else:
        logger.error(f"❌ Step 2: Data Reading and Validation failed: {result.get('error', 'Unknown error')}")
    return result

async def run_step(symbol: str, exchange: str, timeframe: str, data_dir: str=None, **kwargs) -> bool:
    """Standard entry point for Step 2: Data Reading and Validation."""
    result = await run_step_enhanced(symbol, exchange, timeframe, data_dir, **kwargs)
    return result['success']
if __name__ == '__main__':

    async def test() -> None:
        test_symbol = 'TEST_SYMBOL'
        test_exchange = 'TEST_EXCHANGE'
        test_timeframe = '1m'
        result = await run_step_enhanced(symbol=test_symbol, exchange=test_exchange, timeframe=test_timeframe, data_dir=None)
        print(f'Result: {result}')
    asyncio.run(test())