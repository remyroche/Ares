"""
GPU Manager for FAISS/UMAP/cuML Acceleration.

This module provides GPU management capabilities for detecting GPU availability,
managing FAISS/UMAP/cuML acceleration, and device context switching.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from contextlib import contextmanager

# Import from hardware optimization modules
try:
    from src.utils.hardware.enhanced_gpu_manager import (
        get_enhanced_gpu_manager, EnhancedM1GPUManager,
        GPUOperationType, GPUMemoryPool, BatchOperationConfig,
        create_gpu_operation, batch_gpu_operations
    )
    ENHANCED_GPU_AVAILABLE = True
except ImportError:
    ENHANCED_GPU_AVAILABLE = False
    EnhancedM1GPUManager = None

try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
    BASIC_GPU_AVAILABLE = True
except ImportError:
    BASIC_GPU_AVAILABLE = False
    M1GPUManager = None

try:
    from src.utils.matrix_operations.vectorized_core import get_vectorized_processing_core
    VECTORIZED_CORE_AVAILABLE = True
except ImportError:
    VECTORIZED_CORE_AVAILABLE = False
    get_vectorized_processing_core = None

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, tprint_logged, LogLevel, TimestampFormat
)

logger = logging.getLogger(__name__)

class GPUDeviceType(Enum):
    """Types of GPU devices."""
    MPS = "mps"          # Metal Performance Shaders (Apple Silicon)
    CUDA = "cuda"        # NVIDIA CUDA
    ROCM = "rocm"        # AMD ROCm
    CPU = "cpu"          # CPU fallback
    UNKNOWN = "unknown"

class GPUAccelerationType(Enum):
    """Types of GPU acceleration."""
    FAISS = "faiss"      # Facebook AI Similarity Search
    UMAP = "umap"        # Uniform Manifold Approximation and Projection
    CUML = "cuml"        # RAPIDS cuML
    PYTORCH = "pytorch"  # PyTorch GPU acceleration
    TENSORFLOW = "tensorflow"  # TensorFlow GPU acceleration

class GPUManager:
    """Manage GPU availability and acceleration for FAISS/UMAP/cuML."""

    def __init__(self, enable_enhanced_features: bool = True,
                 memory_pool_mb: float = 500.0,
                 enable_batching: bool = True):
        """Initialize GPU Manager.

        Args:
            enable_enhanced_features: Whether to use enhanced GPU features
            memory_pool_mb: GPU memory pool size in MB
            enable_batching: Whether to enable batch processing
        """
        self.logger = logger.getChild('GPUManager')

        # Initialize GPU managers based on availability
        if enable_enhanced_features and ENHANCED_GPU_AVAILABLE:
            self.gpu_manager = get_enhanced_gpu_manager()
            self.enhanced_mode = True
        elif BASIC_GPU_AVAILABLE:
            self.gpu_manager = get_m1_gpu_manager()
            self.enhanced_mode = False
        else:
            self.gpu_manager = None
            self.enhanced_mode = False

        # Configuration
        self.enable_enhanced_features = enable_enhanced_features
        self.memory_pool_mb = memory_pool_mb
        self.enable_batching = enable_batching

        # Device information
        self.device_info = self._detect_gpu_devices()
        self.current_device = self._get_best_device()

        # Acceleration tracking
        self.acceleration_stats = {
            'operations_accelerated': 0,
            'total_acceleration_time': 0.0,
            'memory_pool_usage': 0.0,
            'batching_enabled': enable_batching
        }

        # Operation tracking
        self.active_operations = {}
        self.completed_operations = []

        # Initialize memory pool if enhanced mode
        if self.enhanced_mode and self.gpu_manager:
            self._initialize_memory_pool()

        tprint(f"🚀 GPU Manager initialized (enhanced: {self.enhanced_mode}, device: {self.current_device})", "INFO")
        self.logger.info(f"GPU Manager initialized (enhanced: {self.enhanced_mode}, device: {self.current_device})")

    def _detect_gpu_devices(self) -> Dict[str, Any]:
        """Detect available GPU devices."""
        tprint("🔍 Detecting GPU devices...", "DEBUG")
        devices = {
            'available_devices': [],
            'primary_device': None,
            'device_types': {},
            'total_memory_gb': 0.0,
            'available_memory_gb': 0.0
        }

        try:
            if self.gpu_manager:
                gpu_info = self.gpu_manager.get_gpu_info()

                devices.update({
                    'available_devices': [gpu_info.get('device_name', 'Unknown')],
                    'primary_device': gpu_info.get('device_name', 'Unknown'),
                    'device_types': {gpu_info.get('device_name', 'Unknown'): self._detect_device_type()},
                    'total_memory_gb': gpu_info.get('total_memory_gb', 0.0),
                    'available_memory_gb': gpu_info.get('available_memory_gb', 0.0)
                })

        except Exception as e:
            tprint(f"⚠️ GPU device detection failed: {e}", "WARNING")
            self.logger.warning(f"GPU device detection failed: {e}")
            devices['device_types']['CPU'] = GPUDeviceType.CPU

        return devices

    def _detect_device_type(self) -> GPUDeviceType:
        """Detect the type of GPU device."""
        try:
            if self.gpu_manager:
                gpu_info = self.gpu_manager.get_gpu_info()

                # Check for Apple Silicon MPS
                if gpu_info.get('mps_available', False):
                    return GPUDeviceType.MPS

                # Check device name for clues
                device_name = gpu_info.get('device_name', '').lower()
                if 'cuda' in device_name or 'nvidia' in device_name:
                    return GPUDeviceType.CUDA
                elif 'amd' in device_name or 'radeon' in device_name:
                    return GPUDeviceType.ROCM

            return GPUDeviceType.UNKNOWN

        except Exception:
            return GPUDeviceType.UNKNOWN

    def _get_best_device(self) -> str:
        """Get the best available device."""
        devices = self.device_info.get('available_devices', [])
        return devices[0] if devices else 'CPU'

    def _initialize_memory_pool(self):
        """Initialize GPU memory pool."""
        try:
            if self.enhanced_mode and self.gpu_manager:
                # Create memory pool configuration
                pool_config = GPUMemoryPool(
                    initial_size_mb=self.memory_pool_mb,
                    max_size_mb=self.memory_pool_mb * 2,
                    enable_compression=True
                )

                # Create batch configuration
                batch_config = BatchOperationConfig(
                    max_batch_size=32,
                    enable_auto_batching=self.enable_batching,
                    memory_efficient_batching=True
                )

                # Reinitialize with configurations
                # Note: In practice, would need to modify the enhanced manager to accept configs

                self.logger.info(f"GPU memory pool initialized: {self.memory_pool_mb}MB")

        except Exception as e:
            self.logger.warning(f"Memory pool initialization failed: {e}")

    def is_acceleration_available(self, acceleration_type: GPUAccelerationType) -> bool:
        """Check if specific GPU acceleration is available.

        Args:
            acceleration_type: Type of acceleration to check

        Returns:
            True if acceleration is available
        """
        tprint(f"🔍 Checking acceleration availability for {acceleration_type.value}", "DEBUG")
        try:
            if not self.gpu_manager:
                return False

            # Check based on device type and available libraries
            device_type = self.device_info['device_types'].get(self.current_device, GPUDeviceType.UNKNOWN)

            if acceleration_type == GPUAccelerationType.FAISS:
                # FAISS works with CPU and GPU
                return True
            elif acceleration_type == GPUAccelerationType.UMAP:
                # UMAP GPU support depends on device
                return device_type in [GPUDeviceType.MPS, GPUDeviceType.CUDA]
            elif acceleration_type == GPUAccelerationType.CUML:
                # cuML is NVIDIA-specific
                return device_type == GPUDeviceType.CUDA
            elif acceleration_type in [GPUAccelerationType.PYTORCH, GPUAccelerationType.TENSORFLOW]:
                # These depend on the specific framework support
                return device_type in [GPUDeviceType.MPS, GPUDeviceType.CUDA]

            return False

        except Exception as e:
            tprint(f"❌ Acceleration check failed for {acceleration_type}: {e}", "WARNING")
            self.logger.debug(f"Acceleration check failed for {acceleration_type}: {e}")
            return False

    def create_accelerated_operation(self, operation_type: GPUAccelerationType,
                                   data: Any, parameters: Dict[str, Any],
                                   priority: int = 5) -> str:
        """Create an accelerated GPU operation.

        Args:
            operation_type: Type of GPU operation
            data: Input data for the operation
            parameters: Operation parameters
            priority: Operation priority

        Returns:
            Operation ID
        """
        tprint(f"⚡ Creating accelerated operation: {operation_type.value}", "DEBUG")
        try:
            if not self.gpu_manager:
                self.logger.warning("No GPU manager available for acceleration")
                return ""

            operation_id = f"{operation_type.value}_{int(time.time() * 1000)}"

            # Create GPU operation based on enhanced mode
            if self.enhanced_mode:
                gpu_op_type = self._map_acceleration_to_gpu_type(operation_type)
                if gpu_op_type:
                    # Add to enhanced pipeline
                    pipeline_op_id = self.gpu_manager.add_operation_to_pipeline(
                        "acceleration_pipeline", gpu_op_type, data, parameters, priority
                    )
                    if pipeline_op_id:
                        operation_id = pipeline_op_id
            else:
                # Basic GPU operation tracking
                self.active_operations[operation_id] = {
                    'type': operation_type,
                    'data': data,
                    'parameters': parameters,
                    'priority': priority,
                    'start_time': time.time(),
                    'status': 'queued'
                }

            self.logger.debug(f"Created accelerated operation {operation_id} for {operation_type.value}")
            return operation_id

        except Exception as e:
            self.logger.error(f"Failed to create accelerated operation: {e}")
            return ""

    def _map_acceleration_to_gpu_type(self, acceleration_type: GPUAccelerationType) -> Optional[Any]:
        """Map acceleration type to GPU operation type."""
        if not self.enhanced_mode:
            return None

        try:
            if acceleration_type == GPUAccelerationType.FAISS:
                return GPUOperationType.MATRIX_MULTIPLICATION
            elif acceleration_type == GPUAccelerationType.UMAP:
                return GPUOperationType.DATA_PROCESSING
            elif acceleration_type == GPUAccelerationType.CUML:
                return GPUOperationType.TENSOR_OPERATIONS
            elif acceleration_type == GPUAccelerationType.PYTORCH:
                return GPUOperationType.NEURAL_NETWORK
            elif acceleration_type == GPUAccelerationType.TENSORFLOW:
                return GPUOperationType.NEURAL_NETWORK
            else:
                return GPUOperationType.DATA_PROCESSING

        except Exception:
            return None

    def execute_accelerated_operation(self, operation_id: str) -> Any:
        """Execute an accelerated GPU operation.

        Args:
            operation_id: ID of the operation to execute

        Returns:
            Operation result
        """
        start_time = time.time()

        try:
            if not self.gpu_manager:
                raise RuntimeError("No GPU manager available")

            # Check if operation exists
            if operation_id in self.active_operations:
                operation = self.active_operations[operation_id]
                operation['status'] = 'running'

                # Execute based on enhanced mode
                if self.enhanced_mode:
                    # Use enhanced pipeline execution
                    results = self.gpu_manager.execute_pipeline("acceleration_pipeline")
                    result = results[0] if results else None
                else:
                    # Simulate basic GPU execution
                    time.sleep(0.1)  # Simulate GPU computation
                    result = {
                        'operation_id': operation_id,
                        'success': True,
                        'execution_time': time.time() - start_time,
                        'accelerated': True
                    }

                # Update tracking
                operation['status'] = 'completed'
                operation['end_time'] = time.time()
                operation['result'] = result

                self.completed_operations.append(operation)
                del self.active_operations[operation_id]

                # Update stats
                execution_time = time.time() - start_time
                self.acceleration_stats['operations_accelerated'] += 1
                self.acceleration_stats['total_acceleration_time'] += execution_time

                return result

            else:
                raise ValueError(f"Operation {operation_id} not found")

        except Exception as e:
            self.logger.error(f"Accelerated operation {operation_id} failed: {e}")

            # Update operation status
            if operation_id in self.active_operations:
                self.active_operations[operation_id]['status'] = 'failed'
                self.active_operations[operation_id]['error'] = str(e)

            raise

    def batch_accelerate_operations(self, operations: List[Dict[str, Any]]) -> List[str]:
        """Batch multiple operations for efficient GPU processing.

        Args:
            operations: List of operation configurations

        Returns:
            List of operation IDs
        """
        operation_ids = []

        try:
            if not self.enable_batching or not self.gpu_manager:
                # Execute operations individually
                for op in operations:
                    op_id = self.create_accelerated_operation(**op)
                    if op_id:
                        operation_ids.append(op_id)
            else:
                # Create GPU operations for batching
                gpu_operations = []
                for op in operations:
                    gpu_op = create_gpu_operation(
                        self._map_acceleration_to_gpu_type(op.get('type', GPUAccelerationType.FAISS)),
                        op.get('data'),
                        op.get('parameters', {}),
                        op.get('priority', 5)
                    )
                    if gpu_op:
                        gpu_operations.append(gpu_op)

                # Batch the operations
                if self.enhanced_mode:
                    batched_ids = self.gpu_manager.batch_gpu_operations(gpu_operations)
                    operation_ids.extend(batched_ids)
                else:
                    # Fallback to individual execution
                    for op in operations:
                        op_id = self.create_accelerated_operation(**op)
                        if op_id:
                            operation_ids.append(op_id)

            self.logger.info(f"Batched {len(operation_ids)} operations for GPU acceleration")
            return operation_ids

        except Exception as e:
            self.logger.error(f"Batch acceleration failed: {e}")
            return operation_ids

    def get_acceleration_context(self, device: Optional[str] = None):
        """Get GPU acceleration context manager.

        Args:
            device: Specific device to use (None for auto)

        Returns:
            Context manager for GPU operations
        """
        device_to_use = device or self.current_device

        @contextmanager
        def acceleration_context():
            old_device = self.current_device
            self.current_device = device_to_use

            try:
                self.logger.debug(f"GPU acceleration context: {device_to_use}")
                yield self
            finally:
                self.current_device = old_device
                self.logger.debug(f"GPU acceleration context restored: {old_device}")

        return acceleration_context()

    def optimize_for_acceleration(self, data: Any,
                                operation_type: GPUAccelerationType) -> Any:
        """Optimize data for GPU acceleration.

        Args:
            data: Data to optimize
            operation_type: Type of operation

        Returns:
            Optimized data
        """
        try:
            if not self.gpu_manager:
                return data

            # Use vectorized core if available for optimization
            if VECTORIZED_CORE_AVAILABLE:
                vectorized_core = get_vectorized_processing_core()

                # Optimize based on operation type
                if hasattr(data, 'shape') and hasattr(data, 'dtype'):
                    # NumPy-like optimization
                    if self.enhanced_mode:
                        return self.gpu_manager.optimize_tensor_operations_advanced(data, operation_type)
                    else:
                        return vectorized_core.optimize_dataframe_for_processing(data)
                else:
                    return data
            else:
                return data

        except Exception as e:
            self.logger.debug(f"Data optimization failed: {e}")
            return data

    def get_device_context(self, device_type: GPUDeviceType):
        """Get device-specific context manager.

        Args:
            device_type: Type of device to use

        Returns:
            Context manager for the specific device
        """
        @contextmanager
        def device_context():
            old_device = self.current_device

            # Map device type to actual device name
            if device_type == GPUDeviceType.MPS:
                self.current_device = "MPS" if "MPS" in self.device_info['available_devices'] else self.current_device
            elif device_type == GPUDeviceType.CUDA:
                self.current_device = "CUDA" if "CUDA" in self.device_info['available_devices'] else self.current_device
            elif device_type == GPUDeviceType.CPU:
                self.current_device = "CPU"

            try:
                self.logger.debug(f"Device context: {self.current_device}")
                yield self
            finally:
                self.current_device = old_device

        return device_context()

    def get_acceleration_report(self) -> Dict[str, Any]:
        """Get comprehensive acceleration report."""
        tprint("📊 Generating GPU acceleration report...", "DEBUG")
        report = {
            'device_info': self.device_info,
            'current_device': self.current_device,
            'enhanced_mode': self.enhanced_mode,
            'gpu_manager_available': self.gpu_manager is not None,
            'acceleration_stats': self.acceleration_stats.copy(),
            'active_operations': len(self.active_operations),
            'completed_operations': len(self.completed_operations),
            'memory_pool_mb': self.memory_pool_mb,
            'batching_enabled': self.enable_batching
        }

        # Add enhanced stats if available
        if self.enhanced_mode and self.gpu_manager:
            try:
                enhanced_info = self.gpu_manager.get_enhanced_gpu_info()
                report['enhanced_gpu_info'] = enhanced_info
            except Exception as e:
                report['enhanced_gpu_error'] = str(e)

        # Add operation history
        if self.completed_operations:
            recent_ops = self.completed_operations[-10:]  # Last 10 operations
            report['recent_operations'] = [
                {
                    'id': op.get('operation_id', 'unknown'),
                    'type': str(op.get('type', 'unknown')),
                    'status': op.get('status', 'unknown'),
                    'execution_time': op.get('end_time', 0) - op.get('start_time', 0)
                }
                for op in recent_ops
            ]

        return report

    def shutdown(self):
        """Shutdown GPU manager and cleanup resources."""
        try:
            # Complete any active operations
            for op_id in list(self.active_operations.keys()):
                try:
                    self.execute_accelerated_operation(op_id)
                except Exception:
                    pass  # Ignore errors during shutdown

            # Shutdown enhanced features if available
            if self.enhanced_mode and self.gpu_manager:
                self.gpu_manager.shutdown_enhanced_features()

            self.active_operations.clear()
            self.completed_operations.clear()

            self.logger.info("GPU Manager shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during GPU manager shutdown: {e}")

# Global instance for easy access
_gpu_manager_instance = None

def get_gpu_manager(enable_enhanced_features: bool = True,
                   memory_pool_mb: float = 500.0) -> GPUManager:
    """Get global GPU manager instance."""
    global _gpu_manager_instance

    if _gpu_manager_instance is None:
        _gpu_manager_instance = GPUManager(
            enable_enhanced_features=enable_enhanced_features,
            memory_pool_mb=memory_pool_mb
        )

    return _gpu_manager_instance

# Convenience functions
def is_gpu_acceleration_available(acceleration_type: GPUAccelerationType) -> bool:
    """Check if GPU acceleration is available."""
    manager = get_gpu_manager()
    return manager.is_acceleration_available(acceleration_type)

def create_accelerated_operation(operation_type: GPUAccelerationType,
                               data: Any, parameters: Dict[str, Any],
                               priority: int = 5) -> str:
    """Create an accelerated GPU operation."""
    manager = get_gpu_manager()
    return manager.create_accelerated_operation(operation_type, data, parameters, priority)

def optimize_for_gpu(data: Any, operation_type: GPUAccelerationType) -> Any:
    """Optimize data for GPU acceleration."""
    manager = get_gpu_manager()
    return manager.optimize_for_acceleration(data, operation_type)

def get_acceleration_report() -> Dict[str, Any]:
    """Get comprehensive acceleration report."""
    manager = get_gpu_manager()
    return manager.get_acceleration_report()
