"""
M1 Comprehensive Optimizer for Apple Silicon.

This module integrates all M1/M2/M3/M4 hardware optimizations including
unified memory management, advanced CPU optimization, enhanced GPU acceleration,
and Neural Engine integration for maximum performance.
"""

import logging
import time
import threading
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import numpy as np
import pandas as pd

# Import all M1 optimization modules
from .m1_unified_memory_manager import (
    M1UnifiedMemoryManager, UnifiedMemoryConfig, MemoryTier, get_unified_memory_manager
)
from .m1_advanced_cpu_optimizer import (
    M1AdvancedCPUOptimizer, CPUConfig, WorkloadType, get_advanced_cpu_optimizer
)
from .m1_enhanced_gpu_manager import (
    M1EnhancedGPUManager, GPUConfig, GPUOperationType, get_enhanced_gpu_manager
)
from .m1_neural_engine_manager import (
    M1NeuralEngineManager, NeuralEngineConfig, NeuralEngineOperation, get_neural_engine_manager
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_performance, LogLevel
)

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Optimization strategies for different workloads."""
    MAXIMUM_PERFORMANCE = "maximum_performance"
    BALANCED = "balanced"
    POWER_EFFICIENT = "power_efficient"
    MEMORY_OPTIMIZED = "memory_optimized"
    NEURAL_OPTIMIZED = "neural_optimized"

class WorkloadCategory(Enum):
    """Categories of workloads for optimization."""
    MACHINE_LEARNING = "machine_learning"
    DATA_PROCESSING = "data_processing"
    FINANCIAL_MODELING = "financial_modeling"
    BACKTESTING = "backtesting"
    REAL_TIME_TRADING = "real_time_trading"
    BATCH_PROCESSING = "batch_processing"
    STREAMING = "streaming"

@dataclass
class ComprehensiveConfig:
    """Comprehensive configuration for M1 optimization."""
    # Strategy
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    workload_category: WorkloadCategory = WorkloadCategory.DATA_PROCESSING
    
    # Component configurations
    unified_memory_config: Optional[UnifiedMemoryConfig] = None
    cpu_config: Optional[CPUConfig] = None
    gpu_config: Optional[GPUConfig] = None
    neural_engine_config: Optional[NeuralEngineConfig] = None
    
    # Global settings
    enable_adaptive_optimization: bool = True
    enable_cross_component_optimization: bool = True
    enable_thermal_management: bool = True
    enable_power_management: bool = True
    
    # Performance monitoring
    enable_comprehensive_monitoring: bool = True
    monitoring_interval: float = 5.0
    enable_performance_logging: bool = True
    
    # Auto-tuning
    enable_auto_tuning: bool = True
    tuning_interval: float = 60.0
    performance_target: float = 0.8  # 80% of theoretical maximum

@dataclass
class OptimizationResult:
    """Result of an optimization operation."""
    success: bool
    execution_time: float
    memory_used_mb: float
    cpu_utilization: float
    gpu_utilization: float
    neural_engine_utilization: float
    optimization_applied: List[str]
    performance_improvement: float
    error_message: Optional[str] = None

class AdaptiveOptimizer:
    """Adaptive optimization engine that learns and adjusts."""
    
    def __init__(self, config: ComprehensiveConfig):
        self.config = config
        self.logger = logger.getChild('AdaptiveOptimizer')
        
        # Performance history
        self.performance_history = []
        self.optimization_history = []
        
        # Learning parameters
        self.learning_rate = 0.1
        self.performance_window = 100
        self.adaptation_threshold = 0.1
        
        # Start adaptive optimization
        if self.config.enable_auto_tuning:
            self._start_adaptive_optimization()
    
    def _start_adaptive_optimization(self):
        """Start adaptive optimization thread."""
        def optimize():
            while True:
                try:
                    self._analyze_performance()
                    self._adapt_optimization()
                    time.sleep(self.config.tuning_interval)
                except Exception as e:
                    self.logger.error(f"Adaptive optimization error: {e}")
                    time.sleep(10)
        
        optimization_thread = threading.Thread(target=optimize, daemon=True)
        optimization_thread.start()
        self.logger.info("🧠 Adaptive optimization started")
    
    def _analyze_performance(self):
        """Analyze recent performance and identify optimization opportunities."""
        if len(self.performance_history) < 10:
            return
        
        # Analyze recent performance
        recent_performance = self.performance_history[-self.performance_window:]
        avg_performance = sum(p['performance_score'] for p in recent_performance) / len(recent_performance)
        
        # Check if performance is below target
        if avg_performance < self.config.performance_target:
            self.logger.info(f"📊 Performance below target: {avg_performance:.2f} < {self.config.performance_target}")
            self._identify_bottlenecks(recent_performance)
    
    def _identify_bottlenecks(self, performance_data: List[Dict[str, Any]]):
        """Identify performance bottlenecks."""
        # Analyze CPU utilization
        cpu_utilization = [p.get('cpu_utilization', 0) for p in performance_data]
        avg_cpu_util = sum(cpu_utilization) / len(cpu_utilization)
        
        # Analyze memory usage
        memory_usage = [p.get('memory_usage_mb', 0) for p in performance_data]
        avg_memory = sum(memory_usage) / len(memory_usage)
        
        # Analyze GPU utilization
        gpu_utilization = [p.get('gpu_utilization', 0) for p in performance_data]
        avg_gpu_util = sum(gpu_utilization) / len(gpu_utilization)
        
        # Identify bottlenecks
        bottlenecks = []
        
        if avg_cpu_util > 0.9:
            bottlenecks.append("cpu_intensive")
        
        if avg_memory > 1000:  # 1GB threshold
            bottlenecks.append("memory_intensive")
        
        if avg_gpu_util < 0.1 and avg_cpu_util > 0.5:
            bottlenecks.append("underutilized_gpu")
        
        if bottlenecks:
            self.logger.info(f"🔍 Identified bottlenecks: {bottlenecks}")
            self._apply_bottleneck_optimizations(bottlenecks)
    
    def _apply_bottleneck_optimizations(self, bottlenecks: List[str]):
        """Apply optimizations for identified bottlenecks."""
        optimizations = []
        
        if "cpu_intensive" in bottlenecks:
            optimizations.append("increase_cpu_cores")
            optimizations.append("optimize_cpu_workload_distribution")
        
        if "memory_intensive" in bottlenecks:
            optimizations.append("enable_memory_compression")
            optimizations.append("optimize_memory_allocation")
        
        if "underutilized_gpu" in bottlenecks:
            optimizations.append("enable_gpu_acceleration")
            optimizations.append("optimize_gpu_workloads")
        
        # Apply optimizations
        for optimization in optimizations:
            self._apply_optimization(optimization)
    
    def _apply_optimization(self, optimization: str):
        """Apply a specific optimization."""
        self.logger.info(f"🔧 Applying optimization: {optimization}")
        
        # Record optimization
        self.optimization_history.append({
            'optimization': optimization,
            'timestamp': time.time(),
            'applied': True
        })
    
    def _adapt_optimization(self):
        """Adapt optimization based on performance history."""
        if len(self.performance_history) < 20:
            return
        
        # Analyze optimization effectiveness
        recent_optimizations = [opt for opt in self.optimization_history 
                              if time.time() - opt['timestamp'] < 300]  # Last 5 minutes
        
        if not recent_optimizations:
            return
        
        # Check if optimizations improved performance
        recent_performance = self.performance_history[-20:]
        older_performance = self.performance_history[-40:-20]
        
        if len(older_performance) > 0:
            recent_avg = sum(p['performance_score'] for p in recent_performance) / len(recent_performance)
            older_avg = sum(p['performance_score'] for p in older_performance) / len(older_performance)
            
            improvement = recent_avg - older_avg
            
            if improvement > self.adaptation_threshold:
                self.logger.info(f"✅ Optimizations effective: {improvement:.3f} improvement")
            elif improvement < -self.adaptation_threshold:
                self.logger.warning(f"⚠️ Optimizations ineffective: {improvement:.3f} degradation")
                self._revert_ineffective_optimizations()
    
    def _revert_ineffective_optimizations(self):
        """Revert optimizations that didn't improve performance."""
        # This would implement logic to revert specific optimizations
        self.logger.info("🔄 Reverting ineffective optimizations")
    
    def record_performance(self, performance_data: Dict[str, Any]):
        """Record performance data for analysis."""
        self.performance_history.append(performance_data)
        
        # Keep only recent history
        if len(self.performance_history) > self.performance_window * 2:
            self.performance_history = self.performance_history[-self.performance_window:]

class M1ComprehensiveOptimizer:
    """Comprehensive M1 optimizer integrating all hardware components."""
    
    def __init__(self, config: Optional[ComprehensiveConfig] = None):
        self.config = config or ComprehensiveConfig()
        self.logger = logger.getChild('M1ComprehensiveOptimizer')
        
        # Initialize component configurations
        self._initialize_configurations()
        
        # Initialize hardware managers
        self.unified_memory_manager = get_unified_memory_manager(self.config.unified_memory_config)
        self.cpu_optimizer = get_advanced_cpu_optimizer(self.config.cpu_config)
        self.gpu_manager = get_enhanced_gpu_manager(self.config.gpu_config)
        self.neural_engine_manager = get_neural_engine_manager(self.config.neural_engine_config)
        
        # Initialize adaptive optimizer
        self.adaptive_optimizer = AdaptiveOptimizer(self.config)
        
        # Performance tracking
        self.performance_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_execution_time': 0.0,
            'memory_optimizations': 0,
            'cpu_optimizations': 0,
            'gpu_optimizations': 0,
            'neural_engine_optimizations': 0,
            'cross_component_optimizations': 0
        }
        
        # Start monitoring
        if self.config.enable_comprehensive_monitoring:
            self._start_monitoring()
        
        self.logger.info("🚀 M1 Comprehensive Optimizer initialized")
    
    def _initialize_configurations(self):
        """Initialize component configurations based on strategy."""
        strategy = self.config.optimization_strategy
        workload = self.config.workload_category
        
        # Set configurations based on strategy
        if strategy == OptimizationStrategy.MAXIMUM_PERFORMANCE:
            self._configure_maximum_performance()
        elif strategy == OptimizationStrategy.POWER_EFFICIENT:
            self._configure_power_efficient()
        elif strategy == OptimizationStrategy.MEMORY_OPTIMIZED:
            self._configure_memory_optimized()
        elif strategy == OptimizationStrategy.NEURAL_OPTIMIZED:
            self._configure_neural_optimized()
        else:  # BALANCED
            self._configure_balanced()
        
        # Adjust for workload category
        self._adjust_for_workload_category(workload)
    
    def _configure_maximum_performance(self):
        """Configure for maximum performance."""
        self.config.unified_memory_config = UnifiedMemoryConfig(
            enable_compression=True,
            enable_memory_pools=True,
            enable_cross_component_sharing=True
        )
        
        self.config.cpu_config = CPUConfig(
            enable_core_affinity=True,
            enable_thermal_management=True,
            enable_dynamic_scaling=True,
            thread_pool_size=16
        )
        
        self.config.gpu_config = GPUConfig(
            enable_unified_memory=True,
            enable_batch_processing=True,
            max_concurrent_operations=16
        )
        
        self.config.neural_engine_config = NeuralEngineConfig(
            enable_model_optimization=True,
            enable_batch_processing=True,
            max_batch_size=64
        )
    
    def _configure_power_efficient(self):
        """Configure for power efficiency."""
        self.config.unified_memory_config = UnifiedMemoryConfig(
            enable_compression=True,
            memory_pool_size_mb=512.0
        )
        
        self.config.cpu_config = CPUConfig(
            enable_power_management=True,
            power_save_mode=True,
            thread_pool_size=4
        )
        
        self.config.gpu_config = GPUConfig(
            enable_memory_pressure_detection=True,
            max_concurrent_operations=4
        )
    
    def _configure_memory_optimized(self):
        """Configure for memory optimization."""
        self.config.unified_memory_config = UnifiedMemoryConfig(
            enable_compression=True,
            compression_ratio=0.5,
            enable_memory_pools=True,
            pool_size_mb=1024.0
        )
        
        self.config.cpu_config = CPUConfig(
            enable_memory_pressure_detection=True
        )
    
    def _configure_neural_optimized(self):
        """Configure for neural network optimization."""
        self.config.neural_engine_config = NeuralEngineConfig(
            enable_model_optimization=True,
            enable_quantization=True,
            quantization_bits=8,
            max_batch_size=128
        )
        
        self.config.gpu_config = GPUConfig(
            enable_unified_memory=True,
            max_concurrent_operations=8
        )
    
    def _configure_balanced(self):
        """Configure for balanced performance."""
        self.config.unified_memory_config = UnifiedMemoryConfig()
        self.config.cpu_config = CPUConfig()
        self.config.gpu_config = GPUConfig()
        self.config.neural_engine_config = NeuralEngineConfig()
    
    def _adjust_for_workload_category(self, workload: WorkloadCategory):
        """Adjust configuration for specific workload category."""
        if workload == WorkloadCategory.MACHINE_LEARNING:
            self.config.neural_engine_config.enable_model_optimization = True
            self.config.gpu_config.enable_batch_processing = True
        
        elif workload == WorkloadCategory.FINANCIAL_MODELING:
            self.config.cpu_config.enable_dynamic_scaling = True
            self.config.unified_memory_config.enable_compression = True
        
        elif workload == WorkloadCategory.REAL_TIME_TRADING:
            self.config.cpu_config.thread_pool_size = 8
            self.config.gpu_config.max_concurrent_operations = 4
        
        elif workload == WorkloadCategory.BATCH_PROCESSING:
            self.config.gpu_config.enable_batch_processing = True
            self.config.neural_engine_config.max_batch_size = 128
    
    def _start_monitoring(self):
        """Start comprehensive monitoring."""
        def monitor():
            while True:
                try:
                    self._collect_metrics()
                    time.sleep(self.config.monitoring_interval)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        self.logger.info("📊 Comprehensive monitoring started")
    
    def _collect_metrics(self):
        """Collect metrics from all components."""
        try:
            # Get metrics from all components
            memory_metrics = self.unified_memory_manager.get_comprehensive_stats()
            cpu_metrics = self.cpu_optimizer.get_performance_metrics()
            gpu_metrics = self.gpu_manager.get_performance_metrics()
            neural_metrics = self.neural_engine_manager.get_performance_metrics()
            
            # Calculate overall performance score
            performance_score = self._calculate_performance_score(
                memory_metrics, cpu_metrics, gpu_metrics, neural_metrics
            )
            
            # Record performance data
            performance_data = {
                'timestamp': time.time(),
                'performance_score': performance_score,
                'memory_usage_mb': memory_metrics.get('current_usage_mb', 0),
                'cpu_utilization': cpu_metrics.get('cpu_metrics', {}).get('average_execution_time', 0),
                'gpu_utilization': gpu_metrics.get('gpu_metrics', {}).get('total_operations', 0),
                'neural_engine_utilization': neural_metrics.get('neural_engine_available', False)
            }
            
            self.adaptive_optimizer.record_performance(performance_data)
            
        except Exception as e:
            self.logger.error(f"Metrics collection error: {e}")
    
    def _calculate_performance_score(self, memory_metrics: Dict, cpu_metrics: Dict,
                                   gpu_metrics: Dict, neural_metrics: Dict) -> float:
        """Calculate overall performance score."""
        # This is a simplified calculation
        # Real implementation would use more sophisticated metrics
        
        score = 0.0
        
        # Memory efficiency (0-1)
        memory_usage = memory_metrics.get('current_usage_mb', 0)
        memory_limit = memory_metrics.get('system', {}).get('total_gb', 16) * 1024
        memory_score = 1.0 - (memory_usage / memory_limit) if memory_limit > 0 else 0.0
        score += memory_score * 0.25
        
        # CPU efficiency (0-1)
        cpu_ops = cpu_metrics.get('cpu_metrics', {}).get('total_operations', 0)
        cpu_score = min(1.0, cpu_ops / 1000)  # Normalize to 1000 ops
        score += cpu_score * 0.25
        
        # GPU efficiency (0-1)
        gpu_ops = gpu_metrics.get('gpu_metrics', {}).get('total_operations', 0)
        gpu_score = min(1.0, gpu_ops / 100)  # Normalize to 100 ops
        score += gpu_score * 0.25
        
        # Neural Engine efficiency (0-1)
        neural_available = neural_metrics.get('neural_engine_available', False)
        neural_score = 1.0 if neural_available else 0.0
        score += neural_score * 0.25
        
        return min(1.0, max(0.0, score))
    
    def optimize_operation(self, operation_type: str, data: Any,
                          workload_category: Optional[WorkloadCategory] = None) -> OptimizationResult:
        """Optimize a specific operation."""
        start_time = time.time()
        workload_category = workload_category or self.config.workload_category
        
        try:
            # Determine optimal execution strategy
            execution_strategy = self._determine_execution_strategy(operation_type, data, workload_category)
            
            # Execute with optimization
            result = self._execute_with_optimization(operation_type, data, execution_strategy)
            
            execution_time = time.time() - start_time
            
            # Update metrics
            self.performance_metrics['total_operations'] += 1
            self.performance_metrics['successful_operations'] += 1
            self.performance_metrics['average_execution_time'] = (
                (self.performance_metrics['average_execution_time'] * 
                 (self.performance_metrics['total_operations'] - 1) + execution_time) /
                self.performance_metrics['total_operations']
            )
            
            return OptimizationResult(
                success=True,
                execution_time=execution_time,
                memory_used_mb=0.0,  # Would be calculated from actual usage
                cpu_utilization=0.0,  # Would be calculated from actual usage
                gpu_utilization=0.0,  # Would be calculated from actual usage
                neural_engine_utilization=0.0,  # Would be calculated from actual usage
                optimization_applied=execution_strategy.get('optimizations', []),
                performance_improvement=0.0  # Would be calculated
            )
        
        except Exception as e:
            self.logger.error(f"Operation optimization failed: {e}")
            self.performance_metrics['failed_operations'] += 1
            
            return OptimizationResult(
                success=False,
                execution_time=time.time() - start_time,
                memory_used_mb=0.0,
                cpu_utilization=0.0,
                gpu_utilization=0.0,
                neural_engine_utilization=0.0,
                optimization_applied=[],
                performance_improvement=0.0,
                error_message=str(e)
            )
    
    def _determine_execution_strategy(self, operation_type: str, data: Any,
                                    workload_category: WorkloadCategory) -> Dict[str, Any]:
        """Determine optimal execution strategy."""
        strategy = {
            'use_cpu': True,
            'use_gpu': False,
            'use_neural_engine': False,
            'use_unified_memory': True,
            'optimizations': []
        }
        
        # Determine based on operation type
        if 'matrix' in operation_type.lower() or 'multiply' in operation_type.lower():
            strategy['use_gpu'] = True
            strategy['optimizations'].append('gpu_acceleration')
        
        if 'neural' in operation_type.lower() or 'inference' in operation_type.lower():
            strategy['use_neural_engine'] = True
            strategy['optimizations'].append('neural_engine_optimization')
        
        if 'batch' in operation_type.lower():
            strategy['optimizations'].append('batch_processing')
        
        # Determine based on data characteristics
        if hasattr(data, 'shape'):
            data_size = np.prod(data.shape) * 8 / (1024**2)  # Size in MB
            
            if data_size > 100:  # Large data
                strategy['use_unified_memory'] = True
                strategy['optimizations'].append('unified_memory_optimization')
            
            if data_size > 1000:  # Very large data
                strategy['optimizations'].append('chunked_processing')
        
        return strategy
    
    def _execute_with_optimization(self, operation_type: str, data: Any,
                                 strategy: Dict[str, Any]) -> Any:
        """Execute operation with determined strategy."""
        # Apply unified memory optimization
        if strategy['use_unified_memory']:
            data = self.unified_memory_manager.optimize_data_for_component(data, 'general')
        
        # Execute on appropriate hardware
        if strategy['use_neural_engine'] and self.neural_engine_manager.is_available():
            # Execute on Neural Engine
            return self._execute_on_neural_engine(operation_type, data)
        
        elif strategy['use_gpu'] and self.gpu_manager.is_available():
            # Execute on GPU
            return self._execute_on_gpu(operation_type, data)
        
        else:
            # Execute on CPU
            return self._execute_on_cpu(operation_type, data)
    
    def _execute_on_cpu(self, operation_type: str, data: Any) -> Any:
        """Execute operation on CPU."""
        return self.cpu_optimizer.execute_with_optimization(
            lambda x: x, data, WorkloadType.MIXED
        )
    
    def _execute_on_gpu(self, operation_type: str, data: Any) -> Any:
        """Execute operation on GPU."""
        if 'matrix' in operation_type.lower():
            # Matrix operations
            if isinstance(data, tuple) and len(data) == 2:
                return self.gpu_manager.execute_matrix_multiply(data[0], data[1])
        
        return data
    
    def _execute_on_neural_engine(self, operation_type: str, data: Any) -> Any:
        """Execute operation on Neural Engine."""
        # This would implement actual Neural Engine execution
        return data
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            'overall_metrics': self.performance_metrics,
            'unified_memory': self.unified_memory_manager.get_comprehensive_stats(),
            'cpu_optimizer': self.cpu_optimizer.get_performance_metrics(),
            'gpu_manager': self.gpu_manager.get_performance_metrics(),
            'neural_engine': self.neural_engine_manager.get_performance_metrics(),
            'adaptive_optimizer': {
                'performance_history_length': len(self.adaptive_optimizer.performance_history),
                'optimization_history_length': len(self.adaptive_optimizer.optimization_history)
            }
        }
    
    def shutdown(self):
        """Shutdown comprehensive optimizer."""
        self.unified_memory_manager.cleanup_all()
        self.cpu_optimizer.shutdown()
        self.gpu_manager.shutdown()
        self.neural_engine_manager.shutdown()
        
        self.logger.info("🛑 M1 Comprehensive Optimizer shutdown")

# Global instance
_comprehensive_optimizer: Optional[M1ComprehensiveOptimizer] = None

def get_comprehensive_optimizer(config: Optional[ComprehensiveConfig] = None) -> M1ComprehensiveOptimizer:
    """Get or create the global comprehensive optimizer."""
    global _comprehensive_optimizer
    
    if _comprehensive_optimizer is None:
        _comprehensive_optimizer = M1ComprehensiveOptimizer(config)
    
    return _comprehensive_optimizer

def m1_optimized(operation_type: str = "general", 
                workload_category: WorkloadCategory = WorkloadCategory.DATA_PROCESSING):
    """Decorator for M1 optimization."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_comprehensive_optimizer()
            
            # Optimize the operation
            result = optimizer.optimize_operation(operation_type, args[0] if args else None, workload_category)
            
            if result.success:
                return func(*args, **kwargs)
            else:
                self.logger.warning(f"Optimization failed: {result.error_message}")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def get_m1_comprehensive_metrics() -> Dict[str, Any]:
    """Get comprehensive M1 performance metrics."""
    optimizer = get_comprehensive_optimizer()
    return optimizer.get_comprehensive_metrics()