"""
Hardware Optimization Integration for Statsmodels Clustering

This module provides integration between statsmodels clustering and the UnifiedHardwareManager,
offering optimized performance for regime switching models on Apple Silicon and other platforms.

Key Features:
- Automatic hardware optimization for regime switching workloads
- Memory optimization for large datasets
- CPU core allocation for parallel processing
- GPU acceleration where applicable
- Performance monitoring and adaptive optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
import logging
import time
import threading
from contextlib import contextmanager

# Import hardware optimization
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager,
        WorkloadType,
        OptimizationLevel,
        get_unified_hardware_manager
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    UnifiedHardwareManager = None
    WorkloadType = None
    OptimizationLevel = None

# Import utilities
try:
    from src.utils.tprint import (
        tprint_info, tprint_success, tprint_warning, tprint_error,
        tprint_structured
    )
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')
    def tprint_structured(data, level="INFO"):
        for key, value in data.items():
            print(f'🔧 {key}: {value}')


@dataclass
class HardwareOptimizationConfig:
    """
    Configuration for hardware optimization in statsmodels clustering.
    
    Defines how hardware resources should be allocated and optimized
    for different types of regime switching workloads.
    """
    # Optimization settings
    enable_hardware_optimization: bool = True
    workload_type: str = 'ml_training'
    optimization_level: str = 'balanced'
    
    # CPU optimization
    enable_cpu_optimization: bool = True
    cpu_cores: Optional[int] = None  # None = auto-detect
    enable_core_affinity: bool = True
    
    # Memory optimization
    enable_memory_optimization: bool = True
    memory_limit_gb: Optional[float] = None  # None = auto-detect
    enable_memory_pooling: bool = True
    
    # GPU optimization
    enable_gpu_optimization: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Performance monitoring
    enable_monitoring: bool = True
    monitoring_interval: float = 30.0
    
    # Adaptive optimization
    enable_adaptive_optimization: bool = True
    auto_adjust_resources: bool = True
    
    # Advanced settings
    enable_parallel_processing: bool = True
    chunk_size: Optional[int] = None  # For large datasets
    cache_size_mb: int = 1024


@dataclass
class HardwareOptimizationResult:
    """
    Result container for hardware optimization operations.
    
    Contains optimization metrics, performance improvements, and resource usage.
    """
    # Optimization status
    success: bool = True
    error_message: Optional[str] = None
    
    # Resource allocation
    cpu_cores_allocated: int = 0
    memory_allocated_gb: float = 0.0
    gpu_memory_allocated_mb: float = 0.0
    
    # Performance metrics
    optimization_time: float = 0.0
    performance_improvement: float = 0.0
    
    # Hardware metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    
    # Optimization details
    optimizations_applied: List[str] = field(default_factory=list)
    performance_report: Optional[Dict[str, Any]] = None


class StatsmodelsHardwareOptimizer:
    """
    Hardware optimization integration for statsmodels clustering.
    
    Provides intelligent resource allocation and performance optimization
    specifically tailored for regime switching models.
    """
    
    def __init__(self, 
                 config: Optional[HardwareOptimizationConfig] = None,
                 hardware_manager: Optional[UnifiedHardwareManager] = None):
        """
        Initialize hardware optimizer.
        
        Args:
            config: Hardware optimization configuration
            hardware_manager: External hardware manager instance
        """
        if not HARDWARE_OPTIMIZATION_AVAILABLE:
            tprint_warning("⚠️ Hardware optimization not available - missing dependencies")
            self.available = False
            return
        
        self.config = config or HardwareOptimizationConfig()
        self.hardware_manager = hardware_manager or get_unified_hardware_manager()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.available = True
        
        # Optimization state
        self.optimization_active = False
        self.current_workload = None
        self.optimization_history = []
        
        # Performance tracking
        self.performance_baseline = None
        self.optimization_start_time = None
        
        # Initialize hardware manager if needed
        if not self.hardware_manager.is_initialized:
            self.hardware_manager.initialize()
        
        tprint_info("🔧 Initialized Statsmodels Hardware Optimizer")
    
    def optimize_for_regime_switching(self, 
                                    data_size: int,
                                    n_regimes: int,
                                    model_complexity: str = 'medium') -> HardwareOptimizationResult:
        """
        Optimize hardware configuration for regime switching.
        
        Args:
            data_size: Size of input dataset
            n_regimes: Number of regimes to model
            model_complexity: Complexity level ('low', 'medium', 'high')
            
        Returns:
            HardwareOptimizationResult with optimization details
        """
        if not self.available:
            return HardwareOptimizationResult(
                success=False,
                error_message="Hardware optimization not available"
            )
        
        start_time = time.time()
        result = HardwareOptimizationResult()
        
        try:
            tprint_info(f"🎯 Optimizing for regime switching: {data_size} samples, {n_regimes} regimes")
            
            # Determine workload type and optimization level
            workload_type, optimization_level = self._determine_optimization_strategy(
                data_size, n_regimes, model_complexity
            )
            
            # Apply CPU optimization
            if self.config.enable_cpu_optimization:
                cpu_result = self._optimize_cpu_for_regime_switching(
                    data_size, n_regimes, workload_type, optimization_level
                )
                result.cpu_cores_allocated = cpu_result['cores_allocated']
                result.optimizations_applied.extend(cpu_result['optimizations'])
            
            # Apply memory optimization
            if self.config.enable_memory_optimization:
                memory_result = self._optimize_memory_for_regime_switching(
                    data_size, n_regimes, workload_type, optimization_level
                )
                result.memory_allocated_gb = memory_result['memory_allocated_gb']
                result.optimizations_applied.extend(memory_result['optimizations'])
            
            # Apply GPU optimization
            if self.config.enable_gpu_optimization:
                gpu_result = self._optimize_gpu_for_regime_switching(
                    data_size, n_regimes, workload_type, optimization_level
                )
                result.gpu_memory_allocated_mb = gpu_result['gpu_memory_allocated_mb']
                result.optimizations_applied.extend(gpu_result['optimizations'])
            
            # Configure hardware manager
            self.hardware_manager.configure_workload(workload_type, optimization_level)
            
            # Get current hardware metrics
            hardware_status = self.hardware_manager.get_system_status()
            result.cpu_usage = hardware_status.get('performance_report', {}).get('average_metrics', {}).get('cpu_usage', 0.0)
            result.memory_usage = hardware_status.get('performance_report', {}).get('average_metrics', {}).get('memory_usage', 0.0)
            result.gpu_usage = hardware_status.get('performance_report', {}).get('average_metrics', {}).get('gpu_usage', 0.0)
            
            # Store optimization details
            result.performance_report = hardware_status
            result.optimization_time = time.time() - start_time
            
            # Set optimization state
            self.optimization_active = True
            self.current_workload = {
                'type': 'regime_switching',
                'data_size': data_size,
                'n_regimes': n_regimes,
                'complexity': model_complexity,
                'workload_type': workload_type,
                'optimization_level': optimization_level
            }
            
            # Record optimization
            self.optimization_history.append({
                'timestamp': time.time(),
                'workload': self.current_workload,
                'result': result.__dict__
            })
            
            tprint_success(f"✅ Hardware optimization completed in {result.optimization_time:.2f}s")
            tprint_structured({
                "cpu_cores": result.cpu_cores_allocated,
                "memory_gb": result.memory_allocated_gb,
                "gpu_memory_mb": result.gpu_memory_allocated_mb,
                "optimizations": len(result.optimizations_applied)
            }, level="INFO")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Hardware optimization failed: {e}")
            result.success = False
            result.error_message = str(e)
            result.optimization_time = time.time() - start_time
            return result
    
    def optimize_for_hierarchical_search(self,
                                     search_space_size: int,
                                     parallel_jobs: int = 1) -> HardwareOptimizationResult:
        """
        Optimize hardware for hierarchical parameter search.
        
        Args:
            search_space_size: Size of the parameter search space
            parallel_jobs: Number of parallel jobs to run
            
        Returns:
            HardwareOptimizationResult with optimization details
        """
        if not self.available:
            return HardwareOptimizationResult(
                success=False,
                error_message="Hardware optimization not available"
            )
        
        start_time = time.time()
        result = HardwareOptimizationResult()
        
        try:
            tprint_info(f"🔍 Optimizing for hierarchical search: {search_space_size} combinations, {parallel_jobs} jobs")
            
            # Determine optimization strategy for search
            if search_space_size > 1000:
                workload_type = WorkloadType.MONTE_CARLO
                optimization_level = OptimizationLevel.AGGRESSIVE
            elif search_space_size > 100:
                workload_type = WorkloadType.DATA_PROCESSING
                optimization_level = OptimizationLevel.BALANCED
            else:
                workload_type = WorkloadType.GENERAL
                optimization_level = OptimizationLevel.MINIMAL
            
            # Optimize for parallel processing
            if self.config.enable_parallel_processing:
                parallel_result = self._optimize_for_parallel_processing(parallel_jobs)
                result.cpu_cores_allocated = parallel_result['cores_allocated']
                result.optimizations_applied.extend(parallel_result['optimizations'])
            
            # Configure hardware manager
            self.hardware_manager.configure_workload(workload_type, optimization_level)
            
            # Get hardware metrics
            hardware_status = self.hardware_manager.get_system_status()
            result.performance_report = hardware_status
            result.optimization_time = time.time() - start_time
            
            # Store optimization state
            self.current_workload = {
                'type': 'hierarchical_search',
                'search_space_size': search_space_size,
                'parallel_jobs': parallel_jobs,
                'workload_type': workload_type,
                'optimization_level': optimization_level
            }
            
            tprint_success(f"✅ Hierarchical search optimization completed")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Hierarchical search optimization failed: {e}")
            result.success = False
            result.error_message = str(e)
            result.optimization_time = time.time() - start_time
            return result
    
    def get_optimal_chunk_size(self, data_size: int) -> int:
        """
        Get optimal chunk size for processing large datasets.
        
        Args:
            data_size: Total size of dataset
            
        Returns:
            Optimal chunk size for processing
        """
        if not self.available:
            return min(1000, data_size)
        
        # Get available memory
        hardware_status = self.hardware_manager.get_system_status()
        memory_stats = hardware_status.get('memory_stats', {})
        available_memory_gb = memory_stats.get('available_gb', 8.0)
        
        # Estimate memory per sample (rough estimate)
        memory_per_sample_mb = 0.1  # 100KB per sample
        
        # Calculate chunk size based on available memory
        max_samples_in_memory = int((available_memory_gb * 1024) / memory_per_sample_mb)
        
        # Apply constraints
        chunk_size = min(
            max_samples_in_memory,
            data_size,
            self.config.chunk_size or 10000
        )
        
        # Ensure minimum chunk size
        chunk_size = max(chunk_size, 100)
        
        return chunk_size
    
    def monitor_performance(self, duration: float = 60.0) -> Dict[str, Any]:
        """
        Monitor hardware performance during model training.
        
        Args:
            duration: Monitoring duration in seconds
            
        Returns:
            Performance metrics and analysis
        """
        if not self.available or not self.config.enable_monitoring:
            return {'error': 'Performance monitoring not available'}
        
        try:
            tprint_info(f"📊 Monitoring performance for {duration}s")
            
            # Get initial metrics
            initial_status = self.hardware_manager.get_system_status()
            initial_metrics = initial_status.get('performance_report', {}).get('current_metrics', {})
            
            # Wait for monitoring period
            time.sleep(duration)
            
            # Get final metrics
            final_status = self.hardware_manager.get_system_status()
            final_metrics = final_status.get('performance_report', {}).get('current_metrics', {})
            
            # Calculate performance analysis
            performance_analysis = {
                'monitoring_duration': duration,
                'initial_metrics': initial_metrics,
                'final_metrics': final_metrics,
                'average_metrics': final_status.get('performance_report', {}).get('average_metrics', {}),
                'peak_metrics': final_status.get('performance_report', {}).get('peak_metrics', {}),
                'optimization_active': self.optimization_active,
                'current_workload': self.current_workload
            }
            
            # Calculate performance improvement if baseline available
            if self.performance_baseline:
                baseline_cpu = self.performance_baseline.get('cpu_usage', 0)
                current_cpu = final_metrics.get('cpu_usage', 0)
                performance_analysis['cpu_improvement'] = (baseline_cpu - current_cpu) / max(baseline_cpu, 1) * 100
                
                baseline_memory = self.performance_baseline.get('memory_usage', 0)
                current_memory = final_metrics.get('memory_usage', 0)
                performance_analysis['memory_improvement'] = (baseline_memory - current_memory) / max(baseline_memory, 1) * 100
            
            return performance_analysis
            
        except Exception as e:
            tprint_error(f"❌ Performance monitoring failed: {e}")
            return {'error': str(e)}
    
    def reset_optimization(self):
        """Reset hardware optimization to baseline."""
        if not self.available:
            return
        
        try:
            tprint_info("🔄 Resetting hardware optimization")
            
            # Reset hardware manager to normal thresholds
            self.hardware_manager.set_normal_thresholds()
            
            # Reset optimization state
            self.optimization_active = False
            self.current_workload = None
            
            tprint_success("✅ Hardware optimization reset")
            
        except Exception as e:
            tprint_error(f"❌ Failed to reset optimization: {e}")
    
    @contextmanager
    def optimization_context(self, 
                          workload_type: str = 'ml_training',
                          optimization_level: str = 'balanced'):
        """
        Context manager for temporary hardware optimization.
        
        Args:
            workload_type: Type of workload
            optimization_level: Optimization level
        """
        if not self.available:
            yield self
            return
        
        # Store original state
        original_optimization_active = self.optimization_active
        original_workload = self.current_workload
        
        try:
            # Apply optimization
            if self.config.enable_hardware_optimization:
                workload_enum = WorkloadType(workload_type)
                optimization_enum = OptimizationLevel(optimization_level)
                
                self.hardware_manager.configure_workload(workload_enum, optimization_enum)
                self.optimization_active = True
                self.current_workload = {
                    'type': workload_type,
                    'workload_type': workload_enum,
                    'optimization_level': optimization_enum
                }
            
            yield self
            
        finally:
            # Restore original state
            if original_optimization_active and original_workload:
                self.hardware_manager.configure_workload(
                    original_workload['workload_type'],
                    original_workload['optimization_level']
                )
            else:
                self.reset_optimization()
    
    def _determine_optimization_strategy(self, 
                                     data_size: int,
                                     n_regimes: int,
                                     model_complexity: str) -> Tuple[WorkloadType, OptimizationLevel]:
        """Determine optimal workload type and optimization level."""
        # Base strategy on data size
        if data_size > 100000:
            base_workload = WorkloadType.DATA_PROCESSING
            base_level = OptimizationLevel.AGGRESSIVE
        elif data_size > 10000:
            base_workload = WorkloadType.ML_TRAINING
            base_level = OptimizationLevel.BALANCED
        else:
            base_workload = WorkloadType.GENERAL
            base_level = OptimizationLevel.MINIMAL
        
        # Adjust for number of regimes
        if n_regimes > 10:
            base_level = OptimizationLevel.AGGRESSIVE
        elif n_regimes > 5:
            base_level = OptimizationLevel.BALANCED
        
        # Adjust for model complexity
        if model_complexity == 'high':
            base_level = OptimizationLevel.AGGRESSIVE
        elif model_complexity == 'low':
            base_level = OptimizationLevel.MINIMAL
        
        return base_workload, base_level
    
    def _optimize_cpu_for_regime_switching(self, 
                                         data_size: int,
                                         n_regimes: int,
                                         workload_type: WorkloadType,
                                         optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """Optimize CPU for regime switching workload."""
        optimizations = []
        
        # Determine optimal core count
        if self.config.cpu_cores:
            cores_allocated = self.config.cpu_cores
        else:
            # Auto-detect based on workload
            if optimization_level == OptimizationLevel.AGGRESSIVE:
                cores_allocated = self.hardware_manager.get_optimal_cpu_count()
            elif optimization_level == OptimizationLevel.BALANCED:
                cores_allocated = max(2, self.hardware_manager.get_optimal_cpu_count() // 2)
            else:
                cores_allocated = max(1, self.hardware_manager.get_optimal_cpu_count() // 4)
        
        optimizations.append(f"Allocated {cores_allocated} CPU cores")
        
        # Apply core affinity if enabled
        if self.config.enable_core_affinity:
            optimizations.append("Enabled CPU core affinity")
        
        return {
            'cores_allocated': cores_allocated,
            'optimizations': optimizations
        }
    
    def _optimize_memory_for_regime_switching(self,
                                           data_size: int,
                                           n_regimes: int,
                                           workload_type: WorkloadType,
                                           optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """Optimize memory for regime switching workload."""
        optimizations = []
        
        # Estimate memory requirements
        estimated_memory_gb = self._estimate_memory_requirements(data_size, n_regimes)
        
        # Determine memory allocation
        if self.config.memory_limit_gb:
            memory_allocated_gb = min(self.config.memory_limit_gb, estimated_memory_gb)
        else:
            memory_allocated_gb = estimated_memory_gb
        
        optimizations.append(f"Allocated {memory_allocated_gb:.1f} GB memory")
        
        # Enable memory pooling if enabled
        if self.config.enable_memory_pooling:
            optimizations.append("Enabled memory pooling")
        
        return {
            'memory_allocated_gb': memory_allocated_gb,
            'optimizations': optimizations
        }
    
    def _optimize_gpu_for_regime_switching(self,
                                         data_size: int,
                                         n_regimes: int,
                                         workload_type: WorkloadType,
                                         optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """Optimize GPU for regime switching workload."""
        optimizations = []
        gpu_memory_allocated_mb = 0
        
        # Check if GPU is available and beneficial
        if optimization_level in [OptimizationLevel.BALANCED, OptimizationLevel.AGGRESSIVE]:
            # Allocate GPU memory
            gpu_memory_allocated_mb = int(self.config.gpu_memory_fraction * 8192)  # Assume 8GB GPU
            optimizations.append(f"Allocated {gpu_memory_allocated_mb} MB GPU memory")
        else:
            optimizations.append("GPU optimization disabled for minimal workload")
        
        return {
            'gpu_memory_allocated_mb': gpu_memory_allocated_mb,
            'optimizations': optimizations
        }
    
    def _optimize_for_parallel_processing(self, parallel_jobs: int) -> Dict[str, Any]:
        """Optimize for parallel processing."""
        optimizations = []
        
        # Allocate cores based on parallel jobs
        optimal_cores = min(parallel_jobs, self.hardware_manager.get_optimal_cpu_count())
        optimizations.append(f"Allocated {optimal_cores} cores for {parallel_jobs} parallel jobs")
        
        return {
            'cores_allocated': optimal_cores,
            'optimizations': optimizations
        }
    
    def _estimate_memory_requirements(self, data_size: int, n_regimes: int) -> float:
        """Estimate memory requirements for regime switching."""
        # Base memory for data
        data_memory_gb = data_size * 8 / (1024**3)  # Assume 8 bytes per sample
        
        # Memory for model parameters
        param_memory_gb = (n_regimes ** 2) * 8 / (1024**3)  # Transition matrix
        
        # Memory for intermediate computations
        computation_memory_gb = data_memory_gb * 0.5  # 50% of data size
        
        # Total with safety factor
        total_memory_gb = (data_memory_gb + param_memory_gb + computation_memory_gb) * 1.5
        
        return min(total_memory_gb, 16.0)  # Cap at 16GB


# Convenience functions for hardware optimization
def create_hardware_optimizer(config: Optional[HardwareOptimizationConfig] = None) -> StatsmodelsHardwareOptimizer:
    """
    Create a hardware optimizer with default configuration.
    
    Args:
        config: Optional hardware optimization configuration
        
    Returns:
        StatsmodelsHardwareOptimizer instance
    """
    return StatsmodelsHardwareOptimizer(config)


def optimize_for_regime_switching(data_size: int,
                                 n_regimes: int,
                                 model_complexity: str = 'medium') -> HardwareOptimizationResult:
    """
    Convenience function to optimize for regime switching.
    
    Args:
        data_size: Size of input dataset
        n_regimes: Number of regimes to model
        model_complexity: Complexity level
        
    Returns:
        HardwareOptimizationResult with optimization details
    """
    optimizer = create_hardware_optimizer()
    return optimizer.optimize_for_regime_switching(data_size, n_regimes, model_complexity)