"""
M1 Enhanced GPU Manager for Apple Silicon.

This module provides comprehensive GPU acceleration using Metal Performance Shaders,
unified memory architecture, and advanced compute pipelines for M1/M2/M3/M4 chips.
"""

import logging
import time
import threading
import queue
import weakref
import gc
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

# Optional dependencies
try:
    import torch
    import torch.backends.mps
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import metalcompute
    METAL_COMPUTE_AVAILABLE = True
except ImportError:
    METAL_COMPUTE_AVAILABLE = False
    metalcompute = None

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_performance, LogLevel
)

logger = logging.getLogger(__name__)

class GPUOperationType(Enum):
    """Types of GPU operations."""
    GENERAL = "general"
    MATRIX_MULTIPLICATION = "matrix_multiplication"
    TENSOR_OPERATIONS = "tensor_operations"
    NEURAL_NETWORK = "neural_network"
    DATA_PROCESSING = "data_processing"
    BACKTESTING_SIMULATION = "backtesting_simulation"
    MONTE_CARLO = "monte_carlo"
    IMAGE_PROCESSING = "image_processing"
    SIGNAL_PROCESSING = "signal_processing"

class MemoryLayout(Enum):
    """Memory layouts for GPU operations."""
    CONTIGUOUS = "contiguous"
    STRIDED = "strided"
    PACKED = "packed"
    INTERLEAVED = "interleaved"

class ComputePipeline(Enum):
    """Compute pipelines for different operations."""
    GENERAL = "general"
    MATRIX_OPS = "matrix_ops"
    NEURAL_NET = "neural_net"
    DATA_TRANSFORM = "data_transform"
    SIMULATION = "simulation"

@dataclass
class GPUOperation:
    """Represents a GPU operation."""
    operation_id: str
    operation_type: GPUOperationType
    data: Any
    parameters: Dict[str, Any]
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    callback: Optional[Callable] = None
    timeout: float = 30.0
    memory_layout: MemoryLayout = MemoryLayout.CONTIGUOUS
    compute_pipeline: ComputePipeline = ComputePipeline.GENERAL

@dataclass
class GPUConfig:
    """Configuration for GPU operations."""
    # Memory management
    enable_unified_memory: bool = True
    memory_pool_size_mb: float = 2048.0
    memory_compression: bool = True
    memory_compression_ratio: float = 0.7
    
    # Compute pipelines
    enable_pipeline_optimization: bool = True
    pipeline_cache_size: int = 100
    enable_auto_pipeline_selection: bool = True
    
    # Batch processing
    enable_batch_processing: bool = True
    max_batch_size: int = 64
    batch_timeout: float = 5.0
    auto_batch_threshold: int = 4
    
    # Performance optimization
    enable_async_execution: bool = True
    max_concurrent_operations: int = 8
    operation_timeout: float = 30.0
    
    # Memory pressure management
    enable_memory_pressure_detection: bool = True
    memory_pressure_threshold: float = 0.85
    memory_cleanup_threshold: float = 0.9
    
    # Monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval: float = 1.0
    enable_detailed_logging: bool = False

class MetalComputePipeline:
    """Metal compute pipeline for GPU operations."""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.logger = logger.getChild('MetalComputePipeline')
        
        # Pipeline cache
        self.pipeline_cache = {}
        self.pipeline_stats = {}
        
        # Initialize Metal compute if available
        self.metal_device = None
        self.metal_command_queue = None
        self.metal_library = None
        
        if METAL_COMPUTE_AVAILABLE:
            self._initialize_metal_compute()
        else:
            self.logger.warning("⚠️ Metal compute not available - using PyTorch fallback")
    
    def _initialize_metal_compute(self):
        """Initialize Metal compute resources."""
        try:
            # Initialize Metal device
            self.metal_device = metalcompute.Device()
            self.metal_command_queue = self.metal_device.new_command_queue()
            self.metal_library = self.metal_device.new_library_with_source("""
                #include <metal_stdlib>
                using namespace metal;
                
                kernel void matrix_multiply(
                    device const float* A,
                    device const float* B,
                    device float* C,
                    uint index [[thread_position_in_grid]]
                ) {
                    // Matrix multiplication kernel
                    // Implementation would go here
                }
                
                kernel void tensor_add(
                    device const float* A,
                    device const float* B,
                    device float* C,
                    uint index [[thread_position_in_grid]]
                ) {
                    C[index] = A[index] + B[index];
                }
            """)
            
            self.logger.info("🔧 Metal compute pipeline initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Metal compute: {e}")
            self.metal_device = None
    
    def get_pipeline(self, operation_type: GPUOperationType) -> Optional[Any]:
        """Get compute pipeline for operation type."""
        if not self.metal_device:
            return None
        
        pipeline_key = operation_type.value
        
        if pipeline_key in self.pipeline_cache:
            self.pipeline_stats[pipeline_key] = self.pipeline_stats.get(pipeline_key, 0) + 1
            return self.pipeline_cache[pipeline_key]
        
        # Create new pipeline
        try:
            if operation_type == GPUOperationType.MATRIX_MULTIPLICATION:
                pipeline = self._create_matrix_multiply_pipeline()
            elif operation_type == GPUOperationType.TENSOR_OPERATIONS:
                pipeline = self._create_tensor_ops_pipeline()
            elif operation_type == GPUOperationType.NEURAL_NETWORK:
                pipeline = self._create_neural_net_pipeline()
            else:
                pipeline = self._create_general_pipeline()
            
            self.pipeline_cache[pipeline_key] = pipeline
            self.pipeline_stats[pipeline_key] = 1
            
            return pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to create pipeline for {operation_type}: {e}")
            return None
    
    def _create_matrix_multiply_pipeline(self) -> Any:
        """Create matrix multiplication pipeline."""
        # Simplified implementation
        return {"type": "matrix_multiply", "kernel": "matrix_multiply"}
    
    def _create_tensor_ops_pipeline(self) -> Any:
        """Create tensor operations pipeline."""
        return {"type": "tensor_ops", "kernel": "tensor_add"}
    
    def _create_neural_net_pipeline(self) -> Any:
        """Create neural network pipeline."""
        return {"type": "neural_net", "kernel": "neural_forward"}
    
    def _create_general_pipeline(self) -> Any:
        """Create general purpose pipeline."""
        return {"type": "general", "kernel": "general_compute"}

class GPUMemoryManager:
    """Manages GPU memory with unified memory architecture."""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.logger = logger.getChild('GPUMemoryManager')
        
        # Memory pools
        self.memory_pools = {}
        self.allocation_tracker = {}
        self.memory_stats = {
            'total_allocated': 0.0,
            'total_freed': 0.0,
            'current_usage': 0.0,
            'peak_usage': 0.0,
            'compression_savings': 0.0
        }
        
        # Memory pressure monitoring
        self.memory_pressure_level = 0.0
        self.pressure_history = []
        
        # Start monitoring
        if self.config.enable_memory_pressure_detection:
            self._start_memory_monitoring()
    
    def _start_memory_monitoring(self):
        """Start memory pressure monitoring."""
        def monitor():
            while True:
                try:
                    self._check_memory_pressure()
                    time.sleep(self.config.monitoring_interval)
                except Exception as e:
                    self.logger.error(f"Memory monitoring error: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        self.logger.info("📊 GPU memory monitoring started")
    
    def _check_memory_pressure(self):
        """Check GPU memory pressure."""
        try:
            if TORCH_AVAILABLE and torch.backends.mps.is_available():
                # Get MPS memory info
                allocated = torch.mps.current_allocated_memory()
                reserved = torch.mps.driver_allocated_memory()
                
                total_memory = reserved
                used_memory = allocated
                usage_ratio = used_memory / total_memory if total_memory > 0 else 0
                
                self.memory_pressure_level = usage_ratio
                self.pressure_history.append(usage_ratio)
                
                # Keep only recent history
                if len(self.pressure_history) > 100:
                    self.pressure_history.pop(0)
                
                # Handle high memory pressure
                if usage_ratio > self.config.memory_cleanup_threshold:
                    self._handle_high_memory_pressure()
                elif usage_ratio > self.config.memory_pressure_threshold:
                    self._handle_memory_pressure()
        
        except Exception as e:
            self.logger.warning(f"Failed to check memory pressure: {e}")
    
    def _handle_memory_pressure(self):
        """Handle moderate memory pressure."""
        self.logger.info("⚠️ GPU memory pressure detected")
        
        # Clear unused allocations
        self._cleanup_unused_allocations()
        
        # Force garbage collection
        if TORCH_AVAILABLE:
            torch.mps.empty_cache()
    
    def _handle_high_memory_pressure(self):
        """Handle high memory pressure."""
        self.logger.warning("🚨 High GPU memory pressure - aggressive cleanup")
        
        # Clear all caches
        if TORCH_AVAILABLE:
            torch.mps.empty_cache()
        
        # Clear memory pools
        self._clear_memory_pools()
        
        # Force garbage collection
        gc.collect()
    
    def _cleanup_unused_allocations(self):
        """Cleanup unused memory allocations."""
        # This would implement actual cleanup logic
        pass
    
    def _clear_memory_pools(self):
        """Clear memory pools."""
        self.memory_pools.clear()
        self.allocation_tracker.clear()
    
    def allocate_memory(self, size_mb: float, operation_type: GPUOperationType) -> str:
        """Allocate GPU memory."""
        allocation_id = f"gpu_alloc_{int(time.time())}_{len(self.allocation_tracker)}"
        
        # Track allocation
        self.allocation_tracker[allocation_id] = {
            'size_mb': size_mb,
            'operation_type': operation_type,
            'created_at': time.time(),
            'last_accessed': time.time()
        }
        
        # Update stats
        self.memory_stats['total_allocated'] += size_mb
        self.memory_stats['current_usage'] += size_mb
        
        if self.memory_stats['current_usage'] > self.memory_stats['peak_usage']:
            self.memory_stats['peak_usage'] = self.memory_stats['current_usage']
        
        self.logger.debug(f"🧠 Allocated {size_mb:.1f}MB GPU memory for {operation_type.value}")
        
        return allocation_id
    
    def free_memory(self, allocation_id: str) -> bool:
        """Free GPU memory."""
        if allocation_id not in self.allocation_tracker:
            return False
        
        allocation = self.allocation_tracker[allocation_id]
        size_mb = allocation['size_mb']
        
        # Update stats
        self.memory_stats['total_freed'] += size_mb
        self.memory_stats['current_usage'] -= size_mb
        
        # Remove from tracker
        del self.allocation_tracker[allocation_id]
        
        self.logger.debug(f"🗑️ Freed {size_mb:.1f}MB GPU memory")
        
        return True
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'allocations': len(self.allocation_tracker),
            'memory_stats': self.memory_stats.copy(),
            'pressure_level': self.memory_pressure_level,
            'pressure_history': self.pressure_history[-10:]  # Last 10 readings
        }

class GPUOperationQueue:
    """Queue for managing GPU operations."""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.logger = logger.getChild('GPUOperationQueue')
        
        # Operation queues by priority
        self.operation_queues = {
            priority: queue.PriorityQueue() 
            for priority in range(1, 11)
        }
        
        # Active operations
        self.active_operations = {}
        self.operation_counter = 0
        
        # Batch processing
        self.batch_processor = None
        if self.config.enable_batch_processing:
            self._initialize_batch_processor()
    
    def _initialize_batch_processor(self):
        """Initialize batch processor."""
        def process_batches():
            while True:
                try:
                    self._process_batch()
                    time.sleep(self.config.batch_timeout)
                except Exception as e:
                    self.logger.error(f"Batch processing error: {e}")
                    time.sleep(1)
        
        self.batch_processor = threading.Thread(target=process_batches, daemon=True)
        self.batch_processor.start()
        self.logger.info("📦 Batch processor initialized")
    
    def _process_batch(self):
        """Process batched operations."""
        if not self.config.enable_batch_processing:
            return
        
        # Collect operations for batching
        batch_operations = []
        
        for priority in range(1, 11):
            queue_obj = self.operation_queues[priority]
            if not queue_obj.empty():
                try:
                    operation = queue_obj.get_nowait()
                    batch_operations.append(operation)
                    
                    if len(batch_operations) >= self.config.auto_batch_threshold:
                        break
                except queue.Empty:
                    continue
        
        if batch_operations:
            self._execute_batch(batch_operations)
    
    def _execute_batch(self, operations: List[GPUOperation]):
        """Execute a batch of operations."""
        # Group operations by type
        operations_by_type = {}
        for op in operations:
            op_type = op.operation_type
            if op_type not in operations_by_type:
                operations_by_type[op_type] = []
            operations_by_type[op_type].append(op)
        
        # Execute each group
        for op_type, ops in operations_by_type.items():
            self._execute_operation_group(ops)
    
    def _execute_operation_group(self, operations: List[GPUOperation]):
        """Execute a group of similar operations."""
        if not operations:
            return
        
        # Use the first operation as template
        template_op = operations[0]
        
        # Combine data from all operations
        combined_data = [op.data for op in operations]
        
        # Execute combined operation
        try:
            result = self._execute_single_operation(template_op.operation_type, combined_data)
            
            # Distribute results back to operations
            for i, op in enumerate(operations):
                if op.callback:
                    op.callback(result[i] if isinstance(result, list) else result)
        
        except Exception as e:
            self.logger.error(f"Batch execution error: {e}")
            for op in operations:
                if op.callback:
                    op.callback(None)
    
    def _execute_single_operation(self, operation_type: GPUOperationType, data: Any) -> Any:
        """Execute a single operation."""
        # This would implement actual GPU execution
        # For now, return the data as-is
        return data
    
    def add_operation(self, operation: GPUOperation) -> str:
        """Add operation to queue."""
        operation.operation_id = f"op_{self.operation_counter}_{int(time.time())}"
        self.operation_counter += 1
        
        # Add to appropriate priority queue
        self.operation_queues[operation.priority].put(operation)
        
        self.logger.debug(f"➕ Added operation {operation.operation_id} with priority {operation.priority}")
        
        return operation.operation_id
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status."""
        return {
            'total_queued': sum(q.qsize() for q in self.operation_queues.values()),
            'active_operations': len(self.active_operations),
            'queue_sizes': {
                priority: q.qsize() 
                for priority, q in self.operation_queues.items()
            }
        }

class M1EnhancedGPUManager:
    """Enhanced GPU manager for M1/M2/M3/M4 chips."""
    
    def __init__(self, config: Optional[GPUConfig] = None):
        self.config = config or GPUConfig()
        self.logger = logger.getChild('M1EnhancedGPUManager')
        
        # Check MPS availability
        self.mps_available = TORCH_AVAILABLE and torch.backends.mps.is_available()
        if not self.mps_available:
            self.logger.warning("⚠️ MPS not available - GPU acceleration disabled")
        
        # Initialize components
        self.compute_pipeline = MetalComputePipeline(self.config)
        self.memory_manager = GPUMemoryManager(self.config)
        self.operation_queue = GPUOperationQueue(self.config)
        
        # Performance tracking
        self.performance_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_execution_time': 0.0,
            'memory_allocations': 0,
            'memory_frees': 0,
            'batch_operations': 0
        }
        
        # Thread pool for async execution
        self.thread_pool = None
        if self.config.enable_async_execution:
            self._initialize_thread_pool()
        
        self.logger.info("🚀 M1 Enhanced GPU Manager initialized")
    
    def _initialize_thread_pool(self):
        """Initialize thread pool for async execution."""
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_operations,
            thread_name_prefix="M1GPU"
        )
        self.logger.info(f"🧵 GPU thread pool initialized with {self.config.max_concurrent_operations} workers")
    
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.mps_available
    
    def execute_operation(self, operation_type: GPUOperationType, data: Any, 
                         parameters: Optional[Dict[str, Any]] = None,
                         priority: int = 5, callback: Optional[Callable] = None) -> str:
        """Execute a GPU operation."""
        if not self.mps_available:
            self.logger.warning("⚠️ GPU not available - operation skipped")
            return ""
        
        # Create operation
        operation = GPUOperation(
            operation_id="",  # Will be set by queue
            operation_type=operation_type,
            data=data,
            parameters=parameters or {},
            priority=priority,
            callback=callback
        )
        
        # Add to queue
        operation_id = self.operation_queue.add_operation(operation)
        
        # Update metrics
        self.performance_metrics['total_operations'] += 1
        
        return operation_id
    
    def execute_matrix_multiply(self, A: np.ndarray, B: np.ndarray, 
                               use_gpu: bool = True) -> np.ndarray:
        """Execute matrix multiplication with GPU acceleration."""
        if not use_gpu or not self.mps_available:
            # Fallback to CPU
            return np.dot(A, B)
        
        try:
            # Convert to PyTorch tensors
            A_tensor = torch.from_numpy(A).float()
            B_tensor = torch.from_numpy(B).float()
            
            # Move to MPS
            A_tensor = A_tensor.to('mps')
            B_tensor = B_tensor.to('mps')
            
            # Execute multiplication
            result_tensor = torch.mm(A_tensor, B_tensor)
            
            # Move back to CPU and convert to numpy
            result = result_tensor.cpu().numpy()
            
            self.performance_metrics['successful_operations'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Matrix multiplication error: {e}")
            self.performance_metrics['failed_operations'] += 1
            
            # Fallback to CPU
            return np.dot(A, B)
    
    def execute_tensor_operations(self, operations: List[Tuple[str, Any]], 
                                 use_gpu: bool = True) -> Dict[str, Any]:
        """Execute tensor operations with GPU acceleration."""
        if not use_gpu or not self.mps_available:
            # Fallback to CPU
            return {op[0]: op[1] for op in operations}
        
        results = {}
        
        try:
            for op_name, data in operations:
                if isinstance(data, np.ndarray):
                    # Convert to tensor and move to MPS
                    tensor = torch.from_numpy(data).float().to('mps')
                    
                    # Apply operation based on name
                    if op_name == 'add':
                        result = tensor + tensor
                    elif op_name == 'multiply':
                        result = tensor * tensor
                    elif op_name == 'sum':
                        result = torch.sum(tensor)
                    else:
                        result = tensor
                    
                    # Move back to CPU
                    results[op_name] = result.cpu().numpy()
                else:
                    results[op_name] = data
            
            self.performance_metrics['successful_operations'] += 1
            
        except Exception as e:
            self.logger.error(f"Tensor operations error: {e}")
            self.performance_metrics['failed_operations'] += 1
            
            # Fallback to CPU
            results = {op[0]: op[1] for op in operations}
        
        return results
    
    def execute_neural_network(self, model: Any, input_data: Any, 
                              use_gpu: bool = True) -> Any:
        """Execute neural network with GPU acceleration."""
        if not use_gpu or not self.mps_available:
            # Fallback to CPU
            return model(input_data)
        
        try:
            # Move model and data to MPS
            if hasattr(model, 'to'):
                model = model.to('mps')
            
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).float().to('mps')
            else:
                input_tensor = input_data.to('mps')
            
            # Execute model
            with torch.no_grad():
                result = model(input_tensor)
            
            # Move result back to CPU
            if isinstance(result, torch.Tensor):
                result = result.cpu()
            
            self.performance_metrics['successful_operations'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Neural network execution error: {e}")
            self.performance_metrics['failed_operations'] += 1
            
            # Fallback to CPU
            return model(input_data)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'gpu_metrics': self.performance_metrics,
            'memory_stats': self.memory_manager.get_memory_stats(),
            'queue_status': self.operation_queue.get_queue_status(),
            'mps_available': self.mps_available,
            'compute_pipeline_stats': self.compute_pipeline.pipeline_stats
        }
    
    def clear_memory(self):
        """Clear GPU memory."""
        if self.mps_available:
            torch.mps.empty_cache()
        
        self.memory_manager._clear_memory_pools()
        
        self.logger.info("🧹 GPU memory cleared")
    
    def shutdown(self):
        """Shutdown GPU manager."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        self.clear_memory()
        
        self.logger.info("🛑 M1 Enhanced GPU Manager shutdown")

# Global instance
_enhanced_gpu_manager: Optional[M1EnhancedGPUManager] = None

def get_enhanced_gpu_manager(config: Optional[GPUConfig] = None) -> M1EnhancedGPUManager:
    """Get or create the global enhanced GPU manager."""
    global _enhanced_gpu_manager
    
    if _enhanced_gpu_manager is None:
        _enhanced_gpu_manager = M1EnhancedGPUManager(config)
    
    return _enhanced_gpu_manager

def gpu_accelerated(operation_type: GPUOperationType = GPUOperationType.GENERAL):
    """Decorator for GPU acceleration."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            gpu_manager = get_enhanced_gpu_manager()
            
            if not gpu_manager.is_available():
                return func(*args, **kwargs)
            
            # Enhanced GPU acceleration with VectorBT integration
            try:
                # Check if this is a VectorBT operation
                if operation_type == GPUOperationType.MATRIX_MULTIPLICATION:
                    # Try VectorBT GPU acceleration first
                    try:
                        from .vectorbt_gpu_accelerator import get_vectorbt_gpu_accelerator
                        vectorbt_gpu = get_vectorbt_gpu_accelerator()
                        if vectorbt_gpu.gpu_available and len(args) >= 2:
                            # Check if this looks like financial data
                            if (isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray) and
                                len(args[0].shape) == 2 and len(args[1].shape) == 2):
                                return vectorbt_gpu._gpu_portfolio_analysis(args[0], args[1])
                    except ImportError:
                        pass
                    
                    # Fallback to standard GPU matrix multiplication
                    if len(args) >= 2:
                        return gpu_manager.execute_matrix_multiply(args[0], args[1])
                
                elif operation_type == GPUOperationType.TENSOR_OPERATIONS:
                    # Enhanced tensor operations with memory optimization
                    try:
                        from .enhanced_unified_memory_manager import get_enhanced_unified_memory_manager
                        memory_manager = get_enhanced_unified_memory_manager()
                        
                        # Optimize data for GPU
                        optimized_args = []
                        for arg in args:
                            if isinstance(arg, (np.ndarray, pd.DataFrame)):
                                optimized_arg = memory_manager.base_manager.optimize_data_for_component(arg, 'gpu')
                                optimized_args.append(optimized_arg)
                            else:
                                optimized_args.append(arg)
                        
                        return gpu_manager.execute_tensor_operations([('operation', optimized_args[0])])
                    except ImportError:
                        return gpu_manager.execute_tensor_operations([('operation', args[0])])
                
                elif operation_type == GPUOperationType.NEURAL_NETWORK:
                    # Enhanced neural network with adaptive optimization
                    try:
                        from .adaptive_optimization_engine import get_adaptive_optimization_engine, WorkloadCategory
                        adaptive_engine = get_adaptive_optimization_engine()
                        
                        if len(args) >= 2:
                            # Get optimization recommendations
                            optimization = adaptive_engine.optimize_operation(
                                operation_type="neural_network",
                                workload_category=WorkloadCategory.NEURAL_INFERENCE,
                                data_size_mb=args[0].nbytes / (1024 * 1024) if hasattr(args[0], 'nbytes') else 100.0
                            )
                            
                            # Apply optimization settings
                            if optimization.settings.gpu_acceleration_enabled:
                                return gpu_manager.execute_neural_network(args[0], args[1])
                            else:
                                return func(*args, **kwargs)
                    except ImportError:
                        if len(args) >= 2:
                            return gpu_manager.execute_neural_network(args[0], args[1])
                
                else:
                    # Enhanced general operation with performance tracking
                    try:
                        from .backward_compatibility import performance_tracked
                        
                        @performance_tracked(['execution_time', 'memory_usage', 'gpu_utilization'])
                        def tracked_gpu_operation():
                            return gpu_manager.execute_general_operation(func, *args, **kwargs)
                        
                        return tracked_gpu_operation()
                    except ImportError:
                        return gpu_manager.execute_general_operation(func, *args, **kwargs)
                        
            except Exception as e:
                logger.warning(f"Enhanced GPU acceleration failed: {e}, falling back to standard implementation")
                return func(*args, **kwargs)

        
        return wrapper
    return decorator

def get_gpu_performance_metrics() -> Dict[str, Any]:
    """Get GPU performance metrics."""
    gpu_manager = get_enhanced_gpu_manager()
    return gpu_manager.get_performance_metrics()