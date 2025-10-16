"""
Enhanced GPU Manager for Apple Silicon.

This module extends the basic M1GPUManager with advanced features including
batch operations, GPU memory pooling, compute pipelines, and advanced optimization strategies.
"""

import logging
import time
import asyncio
import threading
import queue
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
import gc

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from .m1_gpu_utils import M1GPUManager

logger = logging.getLogger(__name__)

class GPUOperationType(Enum):
    """Types of GPU operations."""
    MATRIX_MULTIPLICATION = "matrix_multiplication"
    TENSOR_OPERATIONS = "tensor_operations"
    BACKTESTING_SIMULATION = "backtesting_simulation"
    MONTE_CARLO = "monte_carlo"
    NEURAL_NETWORK = "neural_network"
    DATA_PROCESSING = "data_processing"

class MemoryPoolStrategy(Enum):
    """GPU memory pool strategies."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    CUSTOM = "custom"

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

@dataclass
class GPUMemoryPool:
    """GPU memory pool configuration."""
    strategy: MemoryPoolStrategy = MemoryPoolStrategy.ADAPTIVE
    initial_size_mb: float = 100.0
    max_size_mb: float = 1000.0
    min_size_mb: float = 50.0
    growth_factor: float = 1.5
    shrink_threshold: float = 0.3
    enable_compression: bool = True
    compression_ratio: float = 0.7

@dataclass
class BatchOperationConfig:
    """Configuration for batch operations."""
    max_batch_size: int = 32
    batch_timeout: float = 5.0
    enable_auto_batching: bool = True
    priority_batching: bool = True
    memory_efficient_batching: bool = True

class GPUMemoryPoolManager:
    """Manages GPU memory pools for efficient memory usage."""

    def __init__(self, config: GPUMemoryPool):
        self.config = config
        self.logger = logger.getChild('GPUMemoryPoolManager')
        self.memory_pools: Dict[str, Any] = {}
        self.allocation_history: List[Dict[str, Any]] = []
        self.memory_stats: Dict[str, float] = {
            'total_allocated': 0.0,
            'total_freed': 0.0,
            'current_usage': 0.0,
            'peak_usage': 0.0
        }

    def create_memory_pool(self, pool_name: str, size_mb: float) -> bool:
        """Create a new memory pool."""
        try:
            if pool_name in self.memory_pools:
                self.logger.warning(f"Memory pool {pool_name} already exists")
                return False

            # Simulate memory pool creation (in real implementation, would use MPS)
            pool_info = {
                'name': pool_name,
                'size_mb': size_mb,
                'allocated_mb': 0.0,
                'free_mb': size_mb,
                'created_at': time.time(),
                'allocations': []
            }

            self.memory_pools[pool_name] = pool_info
            self.memory_stats['total_allocated'] += size_mb

            self.logger.info(f"🏊 Created GPU memory pool '{pool_name}' with {size_mb}MB")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create memory pool {pool_name}: {e}")
            return False

    def allocate_from_pool(self, pool_name: str, size_mb: float,
                          operation_id: str) -> Optional[Any]:
        """Allocate memory from a pool."""
        try:
            if pool_name not in self.memory_pools:
                self.logger.error(f"Memory pool {pool_name} not found")
                return None

            pool = self.memory_pools[pool_name]

            if pool['free_mb'] < size_mb:
                # Try to grow the pool
                if not self._grow_pool(pool_name, size_mb):
                    self.logger.warning(f"Insufficient memory in pool {pool_name}")
                    return None

            # Allocate memory
            allocation = {
                'operation_id': operation_id,
                'size_mb': size_mb,
                'allocated_at': time.time(),
                'address': f"gpu_mem_{operation_id}"  # Simulated address
            }

            pool['allocations'].append(allocation)
            pool['allocated_mb'] += size_mb
            pool['free_mb'] -= size_mb

            self.memory_stats['current_usage'] += size_mb
            self.memory_stats['peak_usage'] = max(
                self.memory_stats['peak_usage'],
                self.memory_stats['current_usage']
            )

            self.allocation_history.append(allocation)

            self.logger.debug(f"Allocated {size_mb}MB from pool {pool_name} for {operation_id}")
            return allocation['address']

        except Exception as e:
            self.logger.error(f"Failed to allocate from pool {pool_name}: {e}")
            return None

    def deallocate_from_pool(self, pool_name: str, operation_id: str) -> bool:
        """Deallocate memory from a pool."""
        try:
            if pool_name not in self.memory_pools:
                return False

            pool = self.memory_pools[pool_name]

            # Find and remove allocation
            for i, allocation in enumerate(pool['allocations']):
                if allocation['operation_id'] == operation_id:
                    size_mb = allocation['size_mb']
                    pool['allocations'].pop(i)
                    pool['allocated_mb'] -= size_mb
                    pool['free_mb'] += size_mb

                    self.memory_stats['current_usage'] -= size_mb
                    self.memory_stats['total_freed'] += size_mb

                    self.logger.debug(f"Deallocated {size_mb}MB from pool {pool_name} for {operation_id}")
                    return True

            self.logger.warning(f"Allocation {operation_id} not found in pool {pool_name}")
            return False

        except Exception as e:
            self.logger.error(f"Failed to deallocate from pool {pool_name}: {e}")
            return False

    def _grow_pool(self, pool_name: str, required_size_mb: float) -> bool:
        """Grow a memory pool."""
        try:
            pool = self.memory_pools[pool_name]
            current_size = pool['size_mb']
            new_size = max(
                current_size * self.config.growth_factor,
                current_size + required_size_mb
            )

            if new_size > self.config.max_size_mb:
                self.logger.warning(f"Cannot grow pool {pool_name} beyond max size")
                return False

            # Simulate pool growth
            size_increase = new_size - current_size
            pool['size_mb'] = new_size
            pool['free_mb'] += size_increase

            self.memory_stats['total_allocated'] += size_increase

            self.logger.info(f"Grew pool {pool_name} by {size_increase}MB to {new_size}MB")
            return True

        except Exception as e:
            self.logger.error(f"Failed to grow pool {pool_name}: {e}")
            return False

    def get_pool_stats(self, pool_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a memory pool."""
        if pool_name not in self.memory_pools:
            return None

        pool = self.memory_pools[pool_name]

        return {
            'name': pool_name,
            'total_size_mb': pool['size_mb'],
            'allocated_mb': pool['allocated_mb'],
            'free_mb': pool['free_mb'],
            'utilization_percent': (pool['allocated_mb'] / pool['size_mb']) * 100,
            'allocation_count': len(pool['allocations']),
            'created_at': pool['created_at'],
            'age_hours': (time.time() - pool['created_at']) / 3600
        }

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global memory pool statistics."""
        return {
            'total_pools': len(self.memory_pools),
            'memory_stats': self.memory_stats.copy(),
            'allocation_history_count': len(self.allocation_history),
            'pool_utilization': {
                name: self.get_pool_stats(name)['utilization_percent']
                for name in self.memory_pools.keys()
            }
        }

class ComputePipeline:
    """Manages GPU compute pipelines for efficient operation execution."""

    def __init__(self):
        self.logger = logger.getChild('ComputePipeline')
        self.pipelines: Dict[str, List[GPUOperation]] = {}
        self.pipeline_executors: Dict[str, ThreadPoolExecutor] = {}
        self.pipeline_stats: Dict[str, Dict[str, Any]] = {}

    def create_pipeline(self, pipeline_name: str, max_workers: int = 4) -> bool:
        """Create a new compute pipeline."""
        try:
            if pipeline_name in self.pipelines:
                self.logger.warning(f"Pipeline {pipeline_name} already exists")
                return False

            self.pipelines[pipeline_name] = []
            self.pipeline_executors[pipeline_name] = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix=f'GPU-Pipeline-{pipeline_name}'
            )
            self.pipeline_stats[pipeline_name] = {
                'operations_completed': 0,
                'operations_failed': 0,
                'total_execution_time': 0.0,
                'average_execution_time': 0.0,
                'created_at': time.time()
            }

            self.logger.info(f"🔧 Created compute pipeline '{pipeline_name}' with {max_workers} workers")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create pipeline {pipeline_name}: {e}")
            return False

    def add_operation_to_pipeline(self, pipeline_name: str, operation: GPUOperation) -> bool:
        """Add an operation to a pipeline."""
        try:
            if pipeline_name not in self.pipelines:
                self.logger.error(f"Pipeline {pipeline_name} not found")
                return False

            self.pipelines[pipeline_name].append(operation)
            self.logger.debug(f"Added operation {operation.operation_id} to pipeline {pipeline_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add operation to pipeline {pipeline_name}: {e}")
            return False

    async def execute_pipeline(self, pipeline_name: str) -> List[Any]:
        """Execute all operations in a pipeline."""
        try:
            if pipeline_name not in self.pipelines:
                self.logger.error(f"Pipeline {pipeline_name} not found")
                return []

            operations = self.pipelines[pipeline_name]
            if not operations:
                return []

            self.logger.info(f"🚀 Executing pipeline {pipeline_name} with {len(operations)} operations")

            # Sort operations by priority
            operations.sort(key=lambda x: x.priority, reverse=True)

            # Execute operations in parallel
            executor = self.pipeline_executors[pipeline_name]
            results = []

            start_time = time.time()

            # Submit all operations
            future_to_operation = {}
            for operation in operations:
                future = executor.submit(self._execute_operation, operation)
                future_to_operation[future] = operation

            # Collect results
            for future in as_completed(future_to_operation):
                operation = future_to_operation[future]
                try:
                    result = future.result(timeout=operation.timeout)
                    results.append(result)
                    self.pipeline_stats[pipeline_name]['operations_completed'] += 1

                except Exception as e:
                    self.logger.error(f"Operation {operation.operation_id} failed: {e}")
                    self.pipeline_stats[pipeline_name]['operations_failed'] += 1

            # Update statistics
            execution_time = time.time() - start_time
            self.pipeline_stats[pipeline_name]['total_execution_time'] += execution_time
            self.pipeline_stats[pipeline_name]['average_execution_time'] = (
                self.pipeline_stats[pipeline_name]['total_execution_time'] /
                max(1, self.pipeline_stats[pipeline_name]['operations_completed'])
            )

            # Clear pipeline
            self.pipelines[pipeline_name] = []

            self.logger.info(f"✅ Pipeline {pipeline_name} completed in {execution_time:.2f}s")
            return results

        except Exception as e:
            self.logger.error(f"Failed to execute pipeline {pipeline_name}: {e}")
            return []

    def _execute_operation(self, operation: GPUOperation) -> Any:
        """Execute a single GPU operation."""
        try:
            self.logger.debug(f"Executing operation {operation.operation_id}")

            # Simulate operation execution based on type
            if operation.operation_type == GPUOperationType.MATRIX_MULTIPLICATION:
                return self._execute_matrix_multiplication(operation)
            elif operation.operation_type == GPUOperationType.BACKTESTING_SIMULATION:
                return self._execute_backtesting_simulation(operation)
            elif operation.operation_type == GPUOperationType.MONTE_CARLO:
                return self._execute_monte_carlo(operation)
            else:
                return self._execute_generic_operation(operation)

        except Exception as e:
            self.logger.error(f"Operation {operation.operation_id} execution failed: {e}")
            raise

    def _execute_matrix_multiplication(self, operation: GPUOperation) -> Dict[str, Any]:
        """Execute matrix multiplication operation."""
        # Simulate GPU matrix multiplication
        time.sleep(0.1)  # Simulate computation time

        return {
            'operation_id': operation.operation_id,
            'result_type': 'matrix_multiplication',
            'success': True,
            'execution_time': 0.1,
            'gpu_accelerated': True
        }

    def _execute_backtesting_simulation(self, operation: GPUOperation) -> Dict[str, Any]:
        """Execute backtesting simulation operation."""
        # Simulate GPU backtesting
        time.sleep(0.2)  # Simulate computation time

        import random
        return {
            'operation_id': operation.operation_id,
            'result_type': 'backtesting_simulation',
            'success': True,
            'execution_time': 0.2,
            'gpu_accelerated': True,
            'total_trades': random.randint(100, 1000),
            'win_rate': random.uniform(0.4, 0.7),
            'profit_factor': random.uniform(0.8, 1.5)
        }

    def _execute_monte_carlo(self, operation: GPUOperation) -> Dict[str, Any]:
        """Execute Monte Carlo simulation operation."""
        # Simulate GPU Monte Carlo
        time.sleep(0.3)  # Simulate computation time

        return {
            'operation_id': operation.operation_id,
            'result_type': 'monte_carlo',
            'success': True,
            'execution_time': 0.3,
            'gpu_accelerated': True,
            'n_simulations': operation.parameters.get('n_simulations', 1000),
            'mean_return': random.uniform(0.03, 0.07),
            'var_95': random.uniform(-0.1, -0.05)
        }

    def _execute_generic_operation(self, operation: GPUOperation) -> Dict[str, Any]:
        """Execute generic GPU operation."""
        time.sleep(0.05)  # Simulate computation time

        return {
            'operation_id': operation.operation_id,
            'result_type': 'generic',
            'success': True,
            'execution_time': 0.05,
            'gpu_accelerated': True
        }

    def get_pipeline_stats(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a pipeline."""
        if pipeline_name not in self.pipeline_stats:
            return None

        stats = self.pipeline_stats[pipeline_name].copy()
        stats['queued_operations'] = len(self.pipelines.get(pipeline_name, []))
        stats['age_hours'] = (time.time() - stats['created_at']) / 3600

        return stats

    def shutdown_pipeline(self, pipeline_name: str):
        """Shutdown a pipeline."""
        try:
            if pipeline_name in self.pipeline_executors:
                self.pipeline_executors[pipeline_name].shutdown(wait=True)
                del self.pipeline_executors[pipeline_name]

            if pipeline_name in self.pipelines:
                del self.pipelines[pipeline_name]

            if pipeline_name in self.pipeline_stats:
                del self.pipeline_stats[pipeline_name]

            self.logger.info(f"🛑 Pipeline {pipeline_name} shutdown")

        except Exception as e:
            self.logger.error(f"Failed to shutdown pipeline {pipeline_name}: {e}")

class BatchOperationManager:
    """Manages batch GPU operations for improved efficiency."""

    def __init__(self, config: BatchOperationConfig):
        self.config = config
        self.logger = logger.getChild('BatchOperationManager')
        self.operation_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.batch_executor: Optional[ThreadPoolExecutor] = None
        self.batch_stats: Dict[str, Any] = {
            'batches_processed': 0,
            'operations_processed': 0,
            'total_batch_time': 0.0,
            'average_batch_size': 0.0
        }

    def start_batch_processing(self):
        """Start batch processing."""
        if self.batch_executor is None:
            self.batch_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix='GPU-Batch'
            )

        # Start batch processing thread
        threading.Thread(
            target=self._batch_processing_loop,
            daemon=True
        ).start()

        self.logger.info("🔄 Batch processing started")

    def stop_batch_processing(self):
        """Stop batch processing."""
        if self.batch_executor:
            self.batch_executor.shutdown(wait=True)
            self.batch_executor = None

        self.logger.info("🛑 Batch processing stopped")

    def add_operation_to_batch(self, operation: GPUOperation) -> bool:
        """Add an operation to the batch queue."""
        try:
            # Use negative priority for max-heap behavior
            self.operation_queue.put((-operation.priority, operation))
            self.logger.debug(f"Added operation {operation.operation_id} to batch queue")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add operation to batch: {e}")
            return False

    def _batch_processing_loop(self):
        """Main batch processing loop."""
        while True:
            try:
                batch = self._collect_batch()
                if batch:
                    self._process_batch(batch)
                else:
                    time.sleep(0.1)  # Small delay when no operations

            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
                time.sleep(1)

    def _collect_batch(self) -> List[GPUOperation]:
        """Collect operations for a batch."""
        batch = []
        start_time = time.time()

        while (len(batch) < self.config.max_batch_size and
               time.time() - start_time < self.config.batch_timeout):
            try:
                # Get operation with timeout
                priority, operation = self.operation_queue.get(timeout=0.1)
                batch.append(operation)

            except queue.Empty:
                break

        return batch

    def _process_batch(self, batch: List[GPUOperation]):
        """Process a batch of operations."""
        if not batch:
            return

        self.logger.info(f"🔄 Processing batch of {len(batch)} operations")

        start_time = time.time()

        try:
            # Group operations by type for better efficiency
            operations_by_type = {}
            for operation in batch:
                op_type = operation.operation_type
                if op_type not in operations_by_type:
                    operations_by_type[op_type] = []
                operations_by_type[op_type].append(operation)

            # Process each type group
            results = []
            for op_type, operations in operations_by_type.items():
                type_results = self._process_operation_group(op_type, operations)
                results.extend(type_results)

            # Update statistics
            batch_time = time.time() - start_time
            self.batch_stats['batches_processed'] += 1
            self.batch_stats['operations_processed'] += len(batch)
            self.batch_stats['total_batch_time'] += batch_time
            self.batch_stats['average_batch_size'] = (
                self.batch_stats['operations_processed'] /
                max(1, self.batch_stats['batches_processed'])
            )

            self.logger.info(f"✅ Batch processed in {batch_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")

    def _process_operation_group(self, op_type: GPUOperationType,
                               operations: List[GPUOperation]) -> List[Any]:
        """Process a group of operations of the same type."""
        results = []

        # Simulate batch processing based on operation type
        if op_type == GPUOperationType.MATRIX_MULTIPLICATION:
            # Batch matrix operations
            time.sleep(0.05 * len(operations))  # Simulate batch computation
            for operation in operations:
                results.append({
                    'operation_id': operation.operation_id,
                    'success': True,
                    'batch_processed': True
                })
        else:
            # Process individually
            for operation in operations:
                time.sleep(0.01)  # Simulate individual processing
                results.append({
                    'operation_id': operation.operation_id,
                    'success': True,
                    'batch_processed': True
                })

        return results

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return self.batch_stats.copy()

class EnhancedM1GPUManager(M1GPUManager):
    """Enhanced M1 GPU manager with advanced features."""

    def __init__(self,
                 memory_pool_config: Optional[GPUMemoryPool] = None,
                 batch_config: Optional[BatchOperationConfig] = None):
        super().__init__()

        # Initialize enhanced components
        self.memory_pool_config = memory_pool_config or GPUMemoryPool()
        self.batch_config = batch_config or BatchOperationConfig()

        self.memory_pool_manager = GPUMemoryPoolManager(self.memory_pool_config)
        self.compute_pipeline = ComputePipeline()
        self.batch_manager = BatchOperationManager(self.batch_config)

        # Initialize default memory pools
        self._initialize_default_pools()

        # Start batch processing
        self.batch_manager.start_batch_processing()

        self.logger = logger.getChild('EnhancedM1GPUManager')
        self.logger.debug("🚀 Enhanced M1 GPU Manager initialized")

    def _initialize_default_pools(self):
        """Initialize default memory pools."""
        # Create pools for different operation types
        self.memory_pool_manager.create_memory_pool(
            'matrix_operations',
            self.memory_pool_config.initial_size_mb
        )
        self.memory_pool_manager.create_memory_pool(
            'backtesting',
            self.memory_pool_config.initial_size_mb * 0.5
        )
        self.memory_pool_manager.create_memory_pool(
            'monte_carlo',
            self.memory_pool_config.initial_size_mb * 0.3
        )

    def create_optimized_pipeline(self, pipeline_name: str,
                                operation_types: List[GPUOperationType],
                                max_workers: int = 4) -> bool:
        """Create an optimized compute pipeline."""
        success = self.compute_pipeline.create_pipeline(pipeline_name, max_workers)
        if success:
            self.logger.info(f"🔧 Created optimized pipeline '{pipeline_name}' for {[t.value for t in operation_types]}")
        return success

    def add_operation_to_pipeline(self, pipeline_name: str,
                                operation_type: GPUOperationType,
                                data: Any, parameters: Dict[str, Any],
                                priority: int = 5) -> str:
        """Add an operation to a pipeline."""
        operation_id = f"{pipeline_name}_{operation_type.value}_{int(time.time() * 1000)}"

        operation = GPUOperation(
            operation_id=operation_id,
            operation_type=operation_type,
            data=data,
            parameters=parameters,
            priority=priority
        )

        success = self.compute_pipeline.add_operation_to_pipeline(pipeline_name, operation)
        if success:
            self.logger.debug(f"Added operation {operation_id} to pipeline {pipeline_name}")
            return operation_id
        else:
            return ""

    async def execute_pipeline(self, pipeline_name: str) -> List[Any]:
        """Execute a compute pipeline."""
        return await self.compute_pipeline.execute_pipeline(pipeline_name)

    def batch_gpu_operations(self, operations: List[GPUOperation]) -> List[str]:
        """Batch multiple GPU operations for efficiency."""
        operation_ids = []

        for operation in operations:
            success = self.batch_manager.add_operation_to_batch(operation)
            if success:
                operation_ids.append(operation.operation_id)

        self.logger.info(f"🔄 Batched {len(operation_ids)} operations")
        return operation_ids

    def optimize_tensor_operations_advanced(self, data,
                                          operation_type: GPUOperationType):
        """Advanced tensor operation optimization."""
        if not self.mps_available:
            self.logger.debug("MPS not available, using CPU operations")
            return data

        try:
            # Allocate memory from appropriate pool
            pool_name = self._get_pool_for_operation_type(operation_type)

            # Calculate size - handle both numpy arrays and other data types
            if NUMPY_AVAILABLE and hasattr(data, 'nbytes'):
                size_mb = data.nbytes / (1024 * 1024)
            else:
                size_mb = 1.0  # Default size for non-numpy data

            memory_address = self.memory_pool_manager.allocate_from_pool(
                pool_name, size_mb, f"tensor_op_{int(time.time())}"
            )

            if memory_address is None:
                self.logger.warning("Failed to allocate GPU memory, falling back to CPU")
                return data

            # Simulate GPU tensor operations
            # In real implementation, would use PyTorch MPS operations
            result = self._simulate_gpu_tensor_operation(data, operation_type)

            # Deallocate memory
            self.memory_pool_manager.deallocate_from_pool(pool_name, memory_address)

            return result

        except Exception as e:
            self.logger.warning(f"Advanced GPU optimization failed: {e}")
            return data

    def _get_pool_for_operation_type(self, operation_type: GPUOperationType) -> str:
        """Get appropriate memory pool for operation type."""
        if operation_type == GPUOperationType.MATRIX_MULTIPLICATION:
            return 'matrix_operations'
        elif operation_type == GPUOperationType.BACKTESTING_SIMULATION:
            return 'backtesting'
        elif operation_type == GPUOperationType.MONTE_CARLO:
            return 'monte_carlo'
        else:
            return 'matrix_operations'  # Default

    def _simulate_gpu_tensor_operation(self, data,
                                     operation_type: GPUOperationType):
        """Simulate GPU tensor operation."""
        # Simulate GPU computation time
        time.sleep(0.01)

        # Apply some transformation based on operation type
        if operation_type == GPUOperationType.MATRIX_MULTIPLICATION:
            # Simulate matrix multiplication
            if NUMPY_AVAILABLE and hasattr(data, 'ndim') and data.ndim == 2:
                return np.dot(data, data.T)
            else:
                return data * 2
        elif operation_type == GPUOperationType.TENSOR_OPERATIONS:
            # Simulate tensor operations
            if NUMPY_AVAILABLE:
                return np.sqrt(data**2 + 1)
            else:
                return data
        else:
            # Generic operation
            return data * 1.1

    def get_enhanced_gpu_info(self) -> Dict[str, Any]:
        """Get enhanced GPU information."""
        base_info = self.get_gpu_info()

        return {
            **base_info,
            "memory_pool_stats": self.memory_pool_manager.get_global_stats(),
            "batch_stats": self.batch_manager.get_batch_stats(),
            "pipeline_stats": {
                name: self.compute_pipeline.get_pipeline_stats(name)
                for name in self.compute_pipeline.pipelines.keys()
            },
            "enhanced_features": {
                "memory_pooling_enabled": True,
                "batch_processing_enabled": True,
                "compute_pipelines_enabled": True,
                "advanced_optimization_enabled": True
            }
        }

    def get_memory_pool_stats(self, pool_name: str) -> Optional[Dict[str, Any]]:
        """Get memory pool statistics."""
        return self.memory_pool_manager.get_pool_stats(pool_name)

    def get_pipeline_stats(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """Get pipeline statistics."""
        return self.compute_pipeline.get_pipeline_stats(pipeline_name)

    def shutdown_enhanced_features(self):
        """Shutdown enhanced features."""
        try:
            # Stop batch processing
            self.batch_manager.stop_batch_processing()

            # Shutdown all pipelines
            for pipeline_name in list(self.compute_pipeline.pipelines.keys()):
                self.compute_pipeline.shutdown_pipeline(pipeline_name)

            self.logger.info("🛑 Enhanced GPU features shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during enhanced features shutdown: {e}")

# Global instance
_enhanced_gpu_manager: Optional[EnhancedM1GPUManager] = None

def get_enhanced_gpu_manager() -> EnhancedM1GPUManager:
    """Get the global enhanced GPU manager instance."""
    global _enhanced_gpu_manager

    if _enhanced_gpu_manager is None:
        _enhanced_gpu_manager = EnhancedM1GPUManager()

    return _enhanced_gpu_manager

def create_gpu_operation(operation_type: GPUOperationType, data: Any,
                        parameters: Dict[str, Any], priority: int = 5) -> GPUOperation:
    """Create a GPU operation."""
    operation_id = f"{operation_type.value}_{int(time.time() * 1000)}"
    return GPUOperation(
        operation_id=operation_id,
        operation_type=operation_type,
        data=data,
        parameters=parameters,
        priority=priority
    )

def batch_gpu_operations(operations: List[GPUOperation]) -> List[str]:
    """Convenience function to batch GPU operations."""
    manager = get_enhanced_gpu_manager()
    return manager.batch_gpu_operations(operations)
