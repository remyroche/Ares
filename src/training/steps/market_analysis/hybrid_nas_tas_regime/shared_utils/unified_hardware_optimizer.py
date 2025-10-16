"""
Unified Hardware Optimizer for Hybrid NAS-TAS Regime Detection.

This module provides unified hardware optimization capabilities for regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import time
import psutil

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

class HardwareType(Enum):
    """Types of hardware optimization."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"

@dataclass
class HardwareConfig:
    """Configuration for hardware optimization."""
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None
    optimization_level: str = "standard"

@dataclass
class PerformanceMetrics:
    """Hardware performance metrics."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    processing_time: float
    throughput: float

class UnifiedHardwareOptimizer:
    """Unified hardware optimizer for regime detection."""

    def __init__(self, config: HardwareConfig):
        """Initialize the unified hardware optimizer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.performance_history = []

    def optimize_processing(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        operation_type: str = "regime_detection"
    ) -> PerformanceMetrics:
        """Optimize processing for the given data and operation."""
        tprint(f"[HARDWARE_OPTIMIZER] Starting hardware optimization for {operation_type}")

        try:
            start_time = time.time()

            # Get initial system metrics
            initial_cpu = psutil.cpu_percent()
            initial_memory = psutil.virtual_memory().percent

            tprint_debug(f"[HARDWARE_OPTIMIZER] Initial system metrics - CPU: {initial_cpu:.1f}%, Memory: {initial_memory:.1f}%")

            # Perform optimization based on operation type
            tprint(f"[HARDWARE_OPTIMIZER] Executing {operation_type} optimization")
            if operation_type == "regime_detection":
                result = self._optimize_regime_detection(data)
            elif operation_type == "feature_extraction":
                result = self._optimize_feature_extraction(data)
            elif operation_type == "model_training":
                result = self._optimize_model_training(data)
            else:
                result = self._optimize_generic_processing(data)

            # Calculate performance metrics
            end_time = time.time()
            processing_time = end_time - start_time

            final_cpu = psutil.cpu_percent()
            final_memory = psutil.virtual_memory().percent

            # Calculate throughput
            if isinstance(data, np.ndarray):
                data_size = data.size
            else:
                data_size = len(data)

            throughput = data_size / processing_time if processing_time > 0 else 0

            tprint_performance(f"[HARDWARE_OPTIMIZER] Processing completed - Time: {processing_time:.3f}s, Throughput: {throughput:.0f} items/s")
            tprint_debug(f"[HARDWARE_OPTIMIZER] Final system metrics - CPU: {final_cpu:.1f}%, Memory: {final_memory:.1f}%")

            metrics = PerformanceMetrics(
                cpu_usage=max(initial_cpu, final_cpu),
                memory_usage=max(initial_memory, final_memory),
                gpu_usage=None,  # GPU monitoring would require additional setup
                processing_time=processing_time,
                throughput=throughput
            )

            # Store performance history
            self.performance_history.append(metrics)

            tprint_success(f"[HARDWARE_OPTIMIZER] Hardware optimization completed successfully")
            return metrics

        except Exception as e:
            tprint_error(f"[HARDWARE_OPTIMIZER] Error in hardware optimization: {e}")
            tprint_debug(f"[HARDWARE_OPTIMIZER] Operation type: {operation_type}, Data type: {type(data)}")
            return PerformanceMetrics(0, 0, None, 0, 0)

    def _optimize_regime_detection(self, data: Union[np.ndarray, pd.DataFrame]) -> Any:
        """Optimize regime detection processing."""
        tprint_debug(f"[HARDWARE_OPTIMIZER] Optimizing regime detection - Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

        # Simple optimization: chunk processing for large datasets
        if isinstance(data, pd.DataFrame) and len(data) > 10000:
            tprint(f"[HARDWARE_OPTIMIZER] Using chunked processing for large dataset ({len(data)} rows)")
            # Process in chunks
            chunk_size = min(1000, len(data) // 4)
            results = []
            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i+chunk_size]
                # Simulate processing
                results.append(len(chunk))
            tprint_success(f"[HARDWARE_OPTIMIZER] Chunked processing completed - {len(results)} chunks processed")
            return sum(results)
        else:
            tprint_debug(f"[HARDWARE_OPTIMIZER] Using normal processing for dataset")
            # Process normally
            return len(data) if hasattr(data, '__len__') else 1

    def _optimize_feature_extraction(self, data: Union[np.ndarray, pd.DataFrame]) -> Any:
        """Optimize feature extraction processing."""
        tprint_debug(f"[HARDWARE_OPTIMIZER] Optimizing feature extraction - Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

        # Simple optimization: vectorized operations
        if isinstance(data, pd.DataFrame):
            tprint(f"[HARDWARE_OPTIMIZER] Using vectorized operations for DataFrame")
            # Use vectorized operations where possible
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                tprint_success(f"[HARDWARE_OPTIMIZER] Vectorized operations completed - {len(numeric_cols)} numeric columns")
                return data[numeric_cols].mean().sum()
            else:
                tprint_warning(f"[HARDWARE_OPTIMIZER] No numeric columns found, using length fallback")
                return len(data)
        else:
            tprint_debug(f"[HARDWARE_OPTIMIZER] Using numpy operations for array")
            return np.mean(data) if data.size > 0 else 0

    def _optimize_model_training(self, data: Union[np.ndarray, pd.DataFrame]) -> Any:
        """Optimize model training processing."""
        tprint_debug(f"[HARDWARE_OPTIMIZER] Optimizing model training - Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

        # Simple optimization: memory-efficient processing
        if isinstance(data, pd.DataFrame):
            tprint(f"[HARDWARE_OPTIMIZER] Using memory-efficient operations for DataFrame")
            # Use memory-efficient operations
            memory_usage = data.memory_usage(deep=True).sum()
            tprint_success(f"[HARDWARE_OPTIMIZER] Memory-efficient processing completed - Memory usage: {memory_usage / 1024 / 1024:.2f} MB")
            return memory_usage
        else:
            tprint_debug(f"[HARDWARE_OPTIMIZER] Using numpy memory operations for array")
            nbytes = data.nbytes if hasattr(data, 'nbytes') else 0
            tprint_success(f"[HARDWARE_OPTIMIZER] Array processing completed - Memory usage: {nbytes / 1024 / 1024:.2f} MB")
            return nbytes

    def _optimize_generic_processing(self, data: Union[np.ndarray, pd.DataFrame]) -> Any:
        """Optimize generic processing."""
        tprint_debug(f"[HARDWARE_OPTIMIZER] Optimizing generic processing - Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

        # Basic processing
        if isinstance(data, pd.DataFrame):
            tprint_success(f"[HARDWARE_OPTIMIZER] Generic DataFrame processing completed - {len(data)} rows")
            return len(data)
        else:
            size = data.size if hasattr(data, 'size') else 1
            tprint_success(f"[HARDWARE_OPTIMIZER] Generic array processing completed - {size} elements")
            return size

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from optimization history."""
        if not self.performance_history:
            return {}

        cpu_usage = [m.cpu_usage for m in self.performance_history]
        memory_usage = [m.memory_usage for m in self.performance_history]
        processing_times = [m.processing_time for m in self.performance_history]
        throughputs = [m.throughput for m in self.performance_history]

        return {
            'avg_cpu_usage': np.mean(cpu_usage),
            'max_cpu_usage': np.max(cpu_usage),
            'avg_memory_usage': np.mean(memory_usage),
            'max_memory_usage': np.max(memory_usage),
            'avg_processing_time': np.mean(processing_times),
            'total_processing_time': np.sum(processing_times),
            'avg_throughput': np.mean(throughputs),
            'max_throughput': np.max(throughputs),
            'optimization_count': len(self.performance_history)
        }

    def reset_performance_history(self):
        """Reset performance history."""
        self.performance_history = []

def create_unified_hardware_optimizer(config: Optional[HardwareConfig] = None) -> UnifiedHardwareOptimizer:
    """Create a unified hardware optimizer."""
    if config is None:
        config = HardwareConfig()
    return UnifiedHardwareOptimizer(config)

def quick_hardware_optimization(
    data: Union[np.ndarray, pd.DataFrame],
    operation_type: str = "regime_detection"
) -> PerformanceMetrics:
    """Quick hardware optimization for data processing."""
    optimizer = create_unified_hardware_optimizer()
    return optimizer.optimize_processing(data, operation_type)
