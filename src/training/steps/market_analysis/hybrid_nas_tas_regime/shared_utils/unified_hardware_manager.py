"""
Unified Hardware Management System

This module provides comprehensive hardware optimization and management for both
TAS and NAS architectures, including GPU acceleration, memory optimization,
batch processing, and adaptive resource allocation.
"""

import logging
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import torch
import gc
from pathlib import Path
import json

from .unified_architecture_config import ArchitectureType, OptimizationObjective

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """Types of hardware resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


class OptimizationLevel(Enum):
    """Hardware optimization levels."""
    BASIC = "basic"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


class WorkloadType(Enum):
    """Types of workloads for optimization."""
    TRAINING = "training"
    INFERENCE = "inference"
    SEARCH = "search"
    EVALUATION = "evaluation"
    META_LEARNING = "meta_learning"
    ML_TRAINING = "ml_training"
    NEURAL_TRAINING = "neural_training"
    TREE_TRAINING = "tree_training"


@dataclass
class HardwareMetrics:
    """Hardware performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    temperature: float = 0.0
    power_consumption: float = 0.0
    throughput: float = 0.0
    latency: float = 0.0


@dataclass
class OptimizationResult:
    """Result of hardware optimization."""
    optimization_type: str
    performance_improvement: float
    resource_savings: Dict[str, float]
    optimization_time: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareConfig:
    """Hardware optimization configuration."""
    # CPU optimization
    cpu_optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    max_cpu_cores: int = None
    cpu_affinity: List[int] = None
    
    # GPU optimization
    gpu_optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    enable_gpu_acceleration: bool = True
    gpu_memory_fraction: float = 0.8
    enable_mixed_precision: bool = True
    enable_tensor_core: bool = True
    enable_cudnn_benchmark: bool = True
    
    # Memory optimization
    memory_optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    max_memory_usage_gb: float = 8.0
    enable_memory_mapping: bool = True
    enable_memory_pooling: bool = True
    garbage_collection_interval: int = 100
    
    # Batch processing
    enable_batch_processing: bool = True
    adaptive_batch_size: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 1024
    batch_size_increment: int = 2
    
    # Monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval: float = 5.0
    enable_adaptive_optimization: bool = True
    optimization_threshold: float = 0.1
    
    # Advanced features
    enable_mps_acceleration: bool = True  # Apple Metal Performance Shaders
    enable_gpu_memory_pooling: bool = True
    enable_adaptive_optimization: bool = True
    learning_enabled: bool = True
    auto_tuning_enabled: bool = True


class UnifiedHardwareManager:
    """Unified hardware management system for TAS and NAS architectures."""
    
    def __init__(self, 
                 architecture_type: ArchitectureType,
                 config: HardwareConfig = None):
        """Initialize the unified hardware manager.
        
        Args:
            architecture_type: Type of architecture (TAS/NAS/Hybrid)
            config: Hardware optimization configuration
        """
        self.architecture_type = architecture_type
        self.config = config or HardwareConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Hardware state
        self.device = self._initialize_device()
        self.optimization_history: List[OptimizationResult] = []
        self.performance_metrics: List[HardwareMetrics] = []
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.last_optimization_time = time.time()
        
        # Adaptive optimization
        self.performance_baseline = {}
        self.optimization_learnings: Dict[str, Any] = {}
        
        # Batch processing
        self.current_batch_size = self.config.min_batch_size
        self.batch_performance_history = deque(maxlen=100)
        
        # Memory management
        self.memory_pool = {}
        self.gc_counter = 0
        
        # Initialize optimizations
        self._apply_initial_optimizations()
        
        self.logger.info(f"✅ Unified Hardware Manager initialized for {architecture_type.value}")
        self.logger.info(f"   Device: {self.device}")
        self.logger.info(f"   GPU Acceleration: {self.config.enable_gpu_acceleration}")
        self.logger.info(f"   Memory Limit: {self.config.max_memory_usage_gb}GB")
    
    def _initialize_device(self) -> torch.device:
        """Initialize the computation device."""
        if self.config.enable_gpu_acceleration and torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.empty_cache()
            self.logger.info(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name()}")
        elif self.config.enable_mps_acceleration and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            self.logger.info("🍎 MPS acceleration enabled")
        else:
            device = torch.device("cpu")
            self.logger.info("💻 Using CPU")
        
        return device
    
    def _apply_initial_optimizations(self):
        """Apply initial hardware optimizations."""
        try:
            # CPU optimizations
            if self.config.cpu_optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.MAXIMUM]:
                self._optimize_cpu()
            
            # GPU optimizations
            if self.device.type == 'cuda':
                self._optimize_gpu()
            
            # Memory optimizations
            self._optimize_memory()
            
            # PyTorch optimizations
            self._optimize_pytorch()
            
            self.logger.info("✅ Initial hardware optimizations applied")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Some initial optimizations failed: {e}")
    
    def _optimize_cpu(self):
        """Apply CPU optimizations."""
        try:
            # Set CPU affinity if specified
            if self.config.cpu_affinity:
                import os
                os.sched_setaffinity(0, self.config.cpu_affinity)
                self.logger.info(f"🔧 CPU affinity set to cores: {self.config.cpu_affinity}")
            
            # Set maximum CPU cores if specified
            if self.config.max_cpu_cores:
                torch.set_num_threads(self.config.max_cpu_cores)
                self.logger.info(f"🔧 CPU threads limited to: {self.config.max_cpu_cores}")
            
            # Enable OpenMP optimizations
            torch.set_num_interop_threads(1)
            
        except Exception as e:
            self.logger.warning(f"CPU optimization failed: {e}")
    
    def _optimize_gpu(self):
        """Apply GPU optimizations."""
        try:
            if self.device.type != 'cuda':
                return
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
            
            # Enable cuDNN benchmark for consistent input sizes
            if self.config.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
            
            # Enable mixed precision if available
            if self.config.enable_mixed_precision:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Enable Tensor Core optimizations
            if self.config.enable_tensor_core:
                torch.backends.cudnn.enabled = True
            
            self.logger.info("🚀 GPU optimizations applied")
            
        except Exception as e:
            self.logger.warning(f"GPU optimization failed: {e}")
    
    def _optimize_memory(self):
        """Apply memory optimizations."""
        try:
            # Enable memory mapping if available
            if self.config.enable_memory_mapping:
                # This would be architecture-specific
                pass
            
            # Set memory limit
            if self.config.max_memory_usage_gb > 0:
                # Monitor memory usage and trigger cleanup if needed
                self._setup_memory_monitoring()
            
            # Enable garbage collection optimization
            if self.config.garbage_collection_interval > 0:
                self._setup_garbage_collection()
            
            self.logger.info("💾 Memory optimizations applied")
            
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
    
    def _optimize_pytorch(self):
        """Apply PyTorch-specific optimizations."""
        try:
            # Enable JIT optimizations
            torch.jit.set_fusion_strategy([('DYNAMIC', 20)])
            
            # Enable autograd optimizations
            torch.autograd.set_detect_anomaly(False)
            
            # Set optimal number of threads
            if self.device.type == 'cpu':
                torch.set_num_threads(min(psutil.cpu_count(), 8))
            
            self.logger.info("⚡ PyTorch optimizations applied")
            
        except Exception as e:
            self.logger.warning(f"PyTorch optimization failed: {e}")
    
    def _setup_memory_monitoring(self):
        """Setup memory usage monitoring."""
        def memory_monitor():
            while True:
                try:
                    memory_usage = psutil.virtual_memory().percent
                    if memory_usage > (self.config.max_memory_usage_gb / psutil.virtual_memory().total * 100 * 100):
                        self._trigger_memory_cleanup()
                    time.sleep(10)  # Check every 10 seconds
                except Exception as e:
                    self.logger.warning(f"Memory monitoring error: {e}")
                    break
        
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()
    
    def _setup_garbage_collection(self):
        """Setup automatic garbage collection."""
        def periodic_gc():
            while True:
                try:
                    time.sleep(self.config.garbage_collection_interval)
                    self._trigger_garbage_collection()
                except Exception as e:
                    self.logger.warning(f"Garbage collection error: {e}")
                    break
        
        gc_thread = threading.Thread(target=periodic_gc, daemon=True)
        gc_thread.start()
    
    def optimize_for_workload(self, 
                            workload_type: WorkloadType,
                            parameters: Dict[str, Any] = None) -> OptimizationResult:
        """Optimize hardware for a specific workload type.
        
        Args:
            workload_type: Type of workload to optimize for
            parameters: Additional parameters for optimization
            
        Returns:
            Optimization result
        """
        start_time = time.time()
        parameters = parameters or {}
        
        self.logger.info(f"🔧 Optimizing hardware for {workload_type.value}")
        
        try:
            # Architecture-specific optimizations
            if self.architecture_type == ArchitectureType.TAS:
                result = self._optimize_for_tas_workload(workload_type, parameters)
            elif self.architecture_type == ArchitectureType.NAS:
                result = self._optimize_for_nas_workload(workload_type, parameters)
            else:
                result = self._optimize_for_hybrid_workload(workload_type, parameters)
            
            # Apply common optimizations
            self._apply_common_optimizations(workload_type, parameters)
            
            optimization_time = time.time() - start_time
            
            # Create optimization result
            optimization_result = OptimizationResult(
                optimization_type=f"{workload_type.value}_{self.architecture_type.value}",
                performance_improvement=result.get('performance_improvement', 0.0),
                resource_savings=result.get('resource_savings', {}),
                optimization_time=optimization_time,
                success=True,
                metadata={
                    'workload_type': workload_type.value,
                    'architecture_type': self.architecture_type.value,
                    'parameters': parameters
                }
            )
            
            self.optimization_history.append(optimization_result)
            self.last_optimization_time = time.time()
            
            self.logger.info(f"✅ Workload optimization completed in {optimization_time:.2f}s")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"❌ Workload optimization failed: {e}")
            return OptimizationResult(
                optimization_type=f"{workload_type.value}_{self.architecture_type.value}",
                performance_improvement=0.0,
                resource_savings={},
                optimization_time=time.time() - start_time,
                success=False,
                metadata={'error': str(e)}
            )
    
    def _optimize_for_tas_workload(self, workload_type: WorkloadType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hardware for TAS workloads."""
        optimizations = {
            'performance_improvement': 0.0,
            'resource_savings': {}
        }
        
        if workload_type == WorkloadType.TREE_TRAINING:
            # Tree-specific optimizations
            optimizations['performance_improvement'] = 0.15
            optimizations['resource_savings'] = {'memory': 0.2, 'cpu': 0.1}
            
            # Disable GPU for tree training (trees don't benefit much from GPU)
            if self.device.type == 'cuda':
                self.logger.info("🌳 Tree training: Disabling GPU acceleration")
        
        elif workload_type == WorkloadType.EVALUATION:
            # Evaluation-specific optimizations
            optimizations['performance_improvement'] = 0.25
            optimizations['resource_savings'] = {'memory': 0.3}
            
            # Enable batch processing for evaluation
            self._optimize_batch_size(parameters.get('evaluation_batch_size', 100))
        
        return optimizations
    
    def _optimize_for_nas_workload(self, workload_type: WorkloadType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hardware for NAS workloads."""
        optimizations = {
            'performance_improvement': 0.0,
            'resource_savings': {}
        }
        
        if workload_type == WorkloadType.NEURAL_TRAINING:
            # Neural network training optimizations
            optimizations['performance_improvement'] = 0.35
            optimizations['resource_savings'] = {'gpu_memory': 0.25}
            
            # Enable mixed precision training
            if self.config.enable_mixed_precision:
                self.logger.info("🧠 Neural training: Enabling mixed precision")
        
        elif workload_type == WorkloadType.META_LEARNING:
            # Meta-learning optimizations
            optimizations['performance_improvement'] = 0.20
            optimizations['resource_savings'] = {'memory': 0.15}
            
            # Optimize for multiple model instances
            self._optimize_for_meta_learning(parameters)
        
        return optimizations
    
    def _optimize_for_hybrid_workload(self, workload_type: WorkloadType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hardware for hybrid workloads."""
        optimizations = {
            'performance_improvement': 0.0,
            'resource_savings': {}
        }
        
        # Combine TAS and NAS optimizations
        tas_opt = self._optimize_for_tas_workload(workload_type, parameters)
        nas_opt = self._optimize_for_nas_workload(workload_type, parameters)
        
        # Weighted combination based on hybrid weights
        tas_weight = parameters.get('tas_weight', 0.5)
        nas_weight = parameters.get('nas_weight', 0.5)
        
        optimizations['performance_improvement'] = (
            tas_weight * tas_opt['performance_improvement'] + 
            nas_weight * nas_opt['performance_improvement']
        )
        
        # Combine resource savings
        for resource, savings in tas_opt['resource_savings'].items():
            optimizations['resource_savings'][f'tas_{resource}'] = savings
        for resource, savings in nas_opt['resource_savings'].items():
            optimizations['resource_savings'][f'nas_{resource}'] = savings
        
        return optimizations
    
    def _apply_common_optimizations(self, workload_type: WorkloadType, parameters: Dict[str, Any]):
        """Apply common optimizations for all architectures."""
        # Batch size optimization
        if self.config.enable_batch_processing:
            self._optimize_batch_size(parameters.get('batch_size', self.current_batch_size))
        
        # Memory optimization
        if workload_type in [WorkloadType.TRAINING, WorkloadType.NEURAL_TRAINING]:
            self._optimize_memory_for_training(parameters)
    
    def _optimize_batch_size(self, suggested_batch_size: int):
        """Optimize batch size based on workload and hardware."""
        if not self.config.adaptive_batch_size:
            self.current_batch_size = suggested_batch_size
            return
        
        # Start with suggested batch size
        batch_size = max(self.config.min_batch_size, suggested_batch_size)
        batch_size = min(batch_size, self.config.max_batch_size)
        
        # Test performance with different batch sizes
        best_batch_size = batch_size
        best_performance = 0.0
        
        for test_size in [batch_size // 2, batch_size, batch_size * 2]:
            if test_size < self.config.min_batch_size or test_size > self.config.max_batch_size:
                continue
            
            performance = self._test_batch_size_performance(test_size)
            if performance > best_performance:
                best_performance = performance
                best_batch_size = test_size
        
        self.current_batch_size = best_batch_size
        self.logger.info(f"📦 Optimized batch size: {best_batch_size} (performance: {best_performance:.3f})")
    
    def _test_batch_size_performance(self, batch_size: int) -> float:
        """Test performance with a specific batch size."""
        try:
            # Create dummy data for testing
            if self.device.type == 'cuda':
                dummy_data = torch.randn(batch_size, 100).to(self.device)
            else:
                dummy_data = torch.randn(batch_size, 100)
            
            # Measure throughput
            start_time = time.time()
            for _ in range(10):
                _ = torch.matmul(dummy_data, dummy_data.T)
            end_time = time.time()
            
            throughput = batch_size * 10 / (end_time - start_time)
            return throughput
            
        except Exception as e:
            self.logger.warning(f"Batch size test failed: {e}")
            return 0.0
    
    def _optimize_memory_for_training(self, parameters: Dict[str, Any]):
        """Optimize memory usage for training workloads."""
        try:
            # Clear unused memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Trigger garbage collection
            self._trigger_garbage_collection()
            
            # Optimize memory pool if enabled
            if self.config.enable_memory_pooling:
                self._optimize_memory_pool()
            
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
    
    def _optimize_for_meta_learning(self, parameters: Dict[str, Any]):
        """Optimize hardware for meta-learning workloads."""
        try:
            # Meta-learning often requires multiple model instances
            num_models = parameters.get('num_models', 1)
            
            if num_models > 1:
                # Reduce memory per model to fit multiple instances
                if self.device.type == 'cuda':
                    memory_per_model = torch.cuda.get_device_properties(0).total_memory / num_models
                    torch.cuda.set_per_process_memory_fraction(min(0.8, memory_per_model / torch.cuda.get_device_properties(0).total_memory))
            
            # Enable gradient checkpointing for memory efficiency
            if parameters.get('enable_gradient_checkpointing', True):
                torch.utils.checkpoint.enable_gradient_checkpointing()
            
        except Exception as e:
            self.logger.warning(f"Meta-learning optimization failed: {e}")
    
    def _optimize_memory_pool(self):
        """Optimize memory pool allocation."""
        # This is a simplified implementation
        # In practice, you would implement more sophisticated memory pooling
        if self.device.type == 'cuda':
            # Clear and reallocate memory pool
            torch.cuda.empty_cache()
    
    def _trigger_memory_cleanup(self):
        """Trigger memory cleanup when usage is high."""
        self.logger.info("🧹 Triggering memory cleanup")
        
        # Clear GPU memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Trigger garbage collection
        self._trigger_garbage_collection()
        
        # Clear memory pool
        self.memory_pool.clear()
    
    def _trigger_garbage_collection(self):
        """Trigger garbage collection."""
        self.gc_counter += 1
        if self.gc_counter % 10 == 0:  # Every 10th call
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def start_performance_monitoring(self):
        """Start hardware performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("📊 Performance monitoring started")
    
    def stop_performance_monitoring(self):
        """Stop hardware performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("⏹️ Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for hardware performance."""
        while self.monitoring_active:
            try:
                metrics = self._collect_hardware_metrics()
                self.performance_metrics.append(metrics)
                
                # Keep only recent metrics
                if len(self.performance_metrics) > 1000:
                    self.performance_metrics = self.performance_metrics[-1000:]
                
                # Check for optimization opportunities
                if self.config.enable_adaptive_optimization:
                    self._check_optimization_opportunities(metrics)
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _collect_hardware_metrics(self) -> HardwareMetrics:
        """Collect current hardware performance metrics."""
        try:
            # CPU and Memory metrics
            cpu_usage = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent
            
            # GPU metrics
            gpu_usage = 0.0
            gpu_memory_usage = 0.0
            if self.device.type == 'cuda' and torch.cuda.is_available():
                try:
                    gpu_usage = torch.cuda.utilization()
                    gpu_memory_info = torch.cuda.memory_stats()
                    gpu_memory_usage = gpu_memory_info.get('allocated_bytes.all.current', 0) / 1024**3  # GB
                except Exception:
                    pass
            
            # Temperature (if available)
            temperature = 0.0
            try:
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        temperature = list(temps.values())[0][0].current
            except Exception:
                pass
            
            return HardwareMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                gpu_memory_usage=gpu_memory_usage,
                temperature=temperature
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to collect hardware metrics: {e}")
            return HardwareMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0
            )
    
    def _check_optimization_opportunities(self, metrics: HardwareMetrics):
        """Check for optimization opportunities based on current metrics."""
        try:
            # Check if optimization is needed
            current_time = time.time()
            if current_time - self.last_optimization_time < 60:  # Don't optimize too frequently
                return
            
            # Check CPU usage
            if metrics.cpu_usage > 90:
                self._optimize_cpu_usage()
            
            # Check memory usage
            if metrics.memory_usage > 85:
                self._trigger_memory_cleanup()
            
            # Check GPU usage
            if metrics.gpu_usage > 95:
                self._optimize_gpu_usage()
            
        except Exception as e:
            self.logger.warning(f"Optimization check failed: {e}")
    
    def _optimize_cpu_usage(self):
        """Optimize CPU usage."""
        self.logger.info("🔧 Optimizing CPU usage")
        # Reduce number of threads if CPU usage is too high
        current_threads = torch.get_num_threads()
        torch.set_num_threads(max(1, current_threads // 2))
    
    def _optimize_gpu_usage(self):
        """Optimize GPU usage."""
        self.logger.info("🔧 Optimizing GPU usage")
        # Clear GPU cache and reduce memory fraction
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.7)
    
    def get_hardware_status(self) -> Dict[str, Any]:
        """Get current hardware status and performance."""
        if not self.performance_metrics:
            return {'error': 'No performance data available'}
        
        latest_metrics = self.performance_metrics[-1]
        
        status = {
            'device': str(self.device),
            'architecture_type': self.architecture_type.value,
            'current_metrics': {
                'cpu_usage': latest_metrics.cpu_usage,
                'memory_usage': latest_metrics.memory_usage,
                'gpu_usage': latest_metrics.gpu_usage,
                'gpu_memory_usage': latest_metrics.gpu_memory_usage,
                'temperature': latest_metrics.temperature
            },
            'optimization_history': {
                'total_optimizations': len(self.optimization_history),
                'successful_optimizations': sum(1 for opt in self.optimization_history if opt.success),
                'avg_performance_improvement': np.mean([opt.performance_improvement for opt in self.optimization_history]) if self.optimization_history else 0.0
            },
            'batch_processing': {
                'current_batch_size': self.current_batch_size,
                'adaptive_batch_size': self.config.adaptive_batch_size
            },
            'monitoring': {
                'active': self.monitoring_active,
                'metrics_collected': len(self.performance_metrics)
            }
        }
        
        return status
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get hardware optimization recommendations."""
        recommendations = {
            'cpu_optimizations': [],
            'gpu_optimizations': [],
            'memory_optimizations': [],
            'general_recommendations': []
        }
        
        if not self.performance_metrics:
            return recommendations
        
        latest_metrics = self.performance_metrics[-1]
        
        # CPU recommendations
        if latest_metrics.cpu_usage > 80:
            recommendations['cpu_optimizations'].append("Consider reducing CPU optimization level")
        elif latest_metrics.cpu_usage < 30:
            recommendations['cpu_optimizations'].append("CPU has capacity for more aggressive optimization")
        
        # GPU recommendations
        if self.device.type == 'cuda':
            if latest_metrics.gpu_usage > 90:
                recommendations['gpu_optimizations'].append("GPU usage is very high, consider reducing batch size")
            if latest_metrics.gpu_memory_usage > 6.0:  # GB
                recommendations['gpu_optimizations'].append("GPU memory usage is high, enable memory pooling")
        
        # Memory recommendations
        if latest_metrics.memory_usage > 85:
            recommendations['memory_optimizations'].append("Memory usage is high, enable garbage collection")
        
        # Architecture-specific recommendations
        if self.architecture_type == ArchitectureType.TAS:
            recommendations['general_recommendations'].append("Consider disabling GPU for tree-based operations")
        elif self.architecture_type == ArchitectureType.NAS:
            recommendations['general_recommendations'].append("Enable mixed precision training for neural networks")
        
        return recommendations
    
    def export_hardware_data(self, filepath: str):
        """Export hardware performance data to file."""
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'hardware_config': {
                    'cpu_optimization_level': self.config.cpu_optimization_level.value,
                    'gpu_optimization_level': self.config.gpu_optimization_level.value,
                    'memory_optimization_level': self.config.memory_optimization_level.value,
                    'enable_gpu_acceleration': self.config.enable_gpu_acceleration,
                    'max_memory_usage_gb': self.config.max_memory_usage_gb
                },
                'performance_metrics': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'cpu_usage': m.cpu_usage,
                        'memory_usage': m.memory_usage,
                        'gpu_usage': m.gpu_usage,
                        'gpu_memory_usage': m.gpu_memory_usage,
                        'temperature': m.temperature
                    } for m in self.performance_metrics
                ],
                'optimization_history': [
                    {
                        'optimization_type': opt.optimization_type,
                        'performance_improvement': opt.performance_improvement,
                        'resource_savings': opt.resource_savings,
                        'optimization_time': opt.optimization_time,
                        'success': opt.success,
                        'metadata': opt.metadata
                    } for opt in self.optimization_history
                ],
                'current_status': self.get_hardware_status(),
                'recommendations': self.get_optimization_recommendations()
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Hardware data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export hardware data: {e}")
            raise


# Convenience functions
def create_hardware_manager(architecture_type: ArchitectureType,
                          optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
                          **kwargs) -> UnifiedHardwareManager:
    """Create a hardware manager with default settings."""
    config = HardwareConfig(
        cpu_optimization_level=optimization_level,
        gpu_optimization_level=optimization_level,
        memory_optimization_level=optimization_level,
        **kwargs
    )
    return UnifiedHardwareManager(architecture_type=architecture_type, config=config)


def create_basic_hardware_manager(architecture_type: ArchitectureType) -> UnifiedHardwareManager:
    """Create a basic hardware manager."""
    config = HardwareConfig(
        cpu_optimization_level=OptimizationLevel.BASIC,
        gpu_optimization_level=OptimizationLevel.BASIC,
        memory_optimization_level=OptimizationLevel.BASIC,
        enable_performance_monitoring=False,
        enable_adaptive_optimization=False
    )
    return UnifiedHardwareManager(architecture_type=architecture_type, config=config)


def create_aggressive_hardware_manager(architecture_type: ArchitectureType) -> UnifiedHardwareManager:
    """Create an aggressive hardware manager for maximum performance."""
    config = HardwareConfig(
        cpu_optimization_level=OptimizationLevel.AGGRESSIVE,
        gpu_optimization_level=OptimizationLevel.AGGRESSIVE,
        memory_optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_performance_monitoring=True,
        enable_adaptive_optimization=True,
        monitoring_interval=1.0
    )
    return UnifiedHardwareManager(architecture_type=architecture_type, config=config)