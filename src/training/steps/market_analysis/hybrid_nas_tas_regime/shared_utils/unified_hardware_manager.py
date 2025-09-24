"""
Unified Hardware Management System

This module provides comprehensive hardware optimization and management for both
TAS and NAS architectures by leveraging existing hardware utilities from
utils/hardware/ instead of recreating functionality.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import gc
from pathlib import Path
import json

# Import existing hardware utilities
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager as CanonicalHardwareManager,
    WorkloadType as CanonicalWorkloadType,
    OptimizationLevel as CanonicalOptimizationLevel,
    HardwareConfig as CanonicalHardwareConfig,
    PerformanceMetrics as CanonicalPerformanceMetrics
)
from src.utils.hardware.adaptive_optimization_engine import AdaptiveOptimizationEngine
from src.utils.hardware.advanced_cpu_optimizer import AdvancedM1CPUOptimizer
from src.utils.hardware.enhanced_gpu_manager import EnhancedM1GPUManager
from src.utils.hardware.advanced_memory_optimizer import AdvancedM1MemoryOptimizer

from .unified_architecture_config import ArchitectureType, OptimizationObjective

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """Types of hardware resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


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
    cpu_optimization_level: CanonicalOptimizationLevel = CanonicalOptimizationLevel.BALANCED
    max_cpu_cores: int = None
    cpu_affinity: List[int] = None
    
    # GPU optimization
    gpu_optimization_level: CanonicalOptimizationLevel = CanonicalOptimizationLevel.BALANCED
    enable_gpu_acceleration: bool = True
    gpu_memory_fraction: float = 0.8
    enable_mixed_precision: bool = True
    enable_tensor_core: bool = True
    enable_cudnn_benchmark: bool = True
    
    # Memory optimization
    memory_optimization_level: CanonicalOptimizationLevel = CanonicalOptimizationLevel.BALANCED
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
    learning_enabled: bool = True
    auto_tuning_enabled: bool = True


class UnifiedHardwareManager:
    """Unified hardware management system for TAS and NAS architectures using existing hardware utilities."""
    
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
        
        # Initialize existing hardware utilities
        self._initialize_hardware_utilities()
        
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
        
        self.logger.info(f"✅ Unified Hardware Manager initialized for {architecture_type.value}")
        self.logger.info(f"   Using existing hardware utilities from utils/hardware/")
        self.logger.info(f"   Device: {self.device}")
        self.logger.info(f"   GPU Acceleration: {self.config.enable_gpu_acceleration}")
        self.logger.info(f"   Memory Limit: {self.config.max_memory_usage_gb}GB")
    
    def _initialize_hardware_utilities(self):
        """Initialize existing hardware utilities."""
        try:
            # Initialize canonical hardware manager
            canonical_config = self._convert_to_canonical_config()
            self.canonical_hardware_manager = CanonicalHardwareManager(canonical_config)
            
            # Initialize adaptive optimization engine
            self.adaptive_engine = AdaptiveOptimizationEngine(
                hardware_manager=self.canonical_hardware_manager
            )
            
            # Initialize specialized optimizers
            self.cpu_optimizer = AdvancedM1CPUOptimizer()
            self.gpu_manager = EnhancedM1GPUManager()
            self.memory_optimizer = AdvancedM1MemoryOptimizer()
            
            self.logger.info("✅ Existing hardware utilities initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize hardware utilities: {e}")
            raise
    
    def _convert_to_canonical_config(self) -> CanonicalHardwareConfig:
        """Convert unified config to canonical hardware config."""
        return CanonicalHardwareConfig(
            # CPU Configuration
            cpu_optimization_level=self.config.cpu_optimization_level,
            enable_core_affinity=self.config.cpu_affinity is not None,
            enable_thermal_monitoring=True,
            enable_power_management=True,
            
            # GPU Configuration
            gpu_optimization_level=self.config.gpu_optimization_level,
            enable_mps_acceleration=self.config.enable_mps_acceleration,
            enable_gpu_memory_pooling=self.config.enable_gpu_memory_pooling,
            enable_batch_operations=self.config.enable_batch_processing,
            
            # Memory Configuration
            memory_optimization_level=self.config.memory_optimization_level,
            memory_limit_gb=self.config.max_memory_usage_gb,
            enable_memory_pooling=self.config.enable_memory_pooling,
            enable_predictive_allocation=True,
            enable_compression=True,
            
            # Adaptive Configuration
            enable_adaptive_optimization=self.config.enable_adaptive_optimization,
            learning_enabled=self.config.learning_enabled,
            auto_tuning_enabled=self.config.auto_tuning_enabled,
            performance_monitoring_enabled=self.config.enable_performance_monitoring,
            
            # Monitoring Configuration
            monitoring_interval=self.config.monitoring_interval,
            metrics_retention_hours=24,
            alert_thresholds={
                'cpu_usage': 85.0,
                'memory_usage': 90.0,
                'gpu_usage': 80.0,
                'temperature': 85.0
            }
        )
    
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
    
    def optimize_for_workload(self, 
                            workload_type: WorkloadType,
                            parameters: Dict[str, Any] = None) -> OptimizationResult:
        """Optimize hardware for a specific workload type using existing utilities.
        
        Args:
            workload_type: Type of workload to optimize for
            parameters: Additional parameters for optimization
            
        Returns:
            Optimization result
        """
        start_time = time.time()
        parameters = parameters or {}
        
        self.logger.info(f"🔧 Optimizing hardware for {workload_type.value} using existing utilities")
        
        try:
            # Convert workload type to canonical type
            canonical_workload = self._convert_workload_type(workload_type)
            
            # Use existing hardware manager for optimization
            optimization_result = self.canonical_hardware_manager.optimize_for_workload(
                workload_type=canonical_workload,
                parameters=parameters
            )
            
            # Use adaptive optimization engine for advanced optimization
            if self.config.enable_adaptive_optimization:
                adaptive_result = self.adaptive_engine.optimize_workload(
                    workload_type=canonical_workload,
                    parameters=parameters
                )
                
                # Combine results
                optimization_result = self._combine_optimization_results(
                    optimization_result, adaptive_result
                )
            
            optimization_time = time.time() - start_time
            
            # Create unified optimization result
            unified_result = OptimizationResult(
                optimization_type=f"{workload_type.value}_{self.architecture_type.value}",
                performance_improvement=optimization_result.get('performance_improvement', 0.0),
                resource_savings=optimization_result.get('resource_savings', {}),
                optimization_time=optimization_time,
                success=optimization_result.get('success', True),
                metadata={
                    'workload_type': workload_type.value,
                    'architecture_type': self.architecture_type.value,
                    'parameters': parameters,
                    'canonical_result': optimization_result
                }
            )
            
            self.optimization_history.append(unified_result)
            self.last_optimization_time = time.time()
            
            self.logger.info(f"✅ Workload optimization completed in {optimization_time:.2f}s")
            return unified_result
            
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
    
    def _convert_workload_type(self, workload_type: WorkloadType) -> CanonicalWorkloadType:
        """Convert unified workload type to canonical workload type."""
        mapping = {
            WorkloadType.TRAINING: CanonicalWorkloadType.ML_TRAINING,
            WorkloadType.INFERENCE: CanonicalWorkloadType.DATA_PROCESSING,
            WorkloadType.SEARCH: CanonicalWorkloadType.FEATURE_ENGINEERING,
            WorkloadType.EVALUATION: CanonicalWorkloadType.BACKTESTING,
            WorkloadType.META_LEARNING: CanonicalWorkloadType.ML_TRAINING,
            WorkloadType.ML_TRAINING: CanonicalWorkloadType.ML_TRAINING,
            WorkloadType.NEURAL_TRAINING: CanonicalWorkloadType.ML_TRAINING,
            WorkloadType.TREE_TRAINING: CanonicalWorkloadType.FEATURE_ENGINEERING
        }
        return mapping.get(workload_type, CanonicalWorkloadType.GENERAL)
    
    def _combine_optimization_results(self, 
                                    base_result: Dict[str, Any], 
                                    adaptive_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine base and adaptive optimization results."""
        combined = base_result.copy()
        
        # Combine performance improvements
        base_improvement = base_result.get('performance_improvement', 0.0)
        adaptive_improvement = adaptive_result.get('performance_improvement', 0.0)
        combined['performance_improvement'] = max(base_improvement, adaptive_improvement)
        
        # Combine resource savings
        base_savings = base_result.get('resource_savings', {})
        adaptive_savings = adaptive_result.get('resource_savings', {})
        combined['resource_savings'] = {**base_savings, **adaptive_savings}
        
        # Add adaptive metadata
        combined['adaptive_optimization'] = adaptive_result
        
        return combined
    
    def start_performance_monitoring(self):
        """Start hardware performance monitoring using existing utilities."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Use existing hardware manager for monitoring
        self.canonical_hardware_manager.start_monitoring()
        
        # Start custom monitoring thread for unified metrics
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("📊 Performance monitoring started using existing utilities")
    
    def stop_performance_monitoring(self):
        """Stop hardware performance monitoring."""
        self.monitoring_active = False
        
        # Stop canonical hardware manager monitoring
        if hasattr(self.canonical_hardware_manager, 'stop_monitoring'):
            self.canonical_hardware_manager.stop_monitoring()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("⏹️ Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for hardware performance."""
        while self.monitoring_active:
            try:
                # Get metrics from canonical hardware manager
                canonical_metrics = self.canonical_hardware_manager.get_performance_metrics()
                
                # Convert to unified format
                unified_metrics = self._convert_to_unified_metrics(canonical_metrics)
                self.performance_metrics.append(unified_metrics)
                
                # Keep only recent metrics
                if len(self.performance_metrics) > 1000:
                    self.performance_metrics = self.performance_metrics[-1000:]
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _convert_to_unified_metrics(self, canonical_metrics: CanonicalPerformanceMetrics) -> HardwareMetrics:
        """Convert canonical metrics to unified format."""
        return HardwareMetrics(
            timestamp=datetime.fromtimestamp(canonical_metrics.timestamp),
            cpu_usage=canonical_metrics.cpu_usage,
            memory_usage=canonical_metrics.memory_usage,
            gpu_usage=canonical_metrics.gpu_usage,
            temperature=canonical_metrics.temperature,
            power_consumption=canonical_metrics.power_consumption,
            throughput=canonical_metrics.performance_score,
            latency=0.0  # Not available in canonical metrics
        )
    
    def get_hardware_status(self) -> Dict[str, Any]:
        """Get current hardware status and performance."""
        status = {
            'device': str(self.device),
            'architecture_type': self.architecture_type.value,
            'using_existing_utilities': True,
            'canonical_hardware_status': {},
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
        
        # Get status from canonical hardware manager
        try:
            canonical_status = self.canonical_hardware_manager.get_status()
            status['canonical_hardware_status'] = canonical_status
        except Exception as e:
            self.logger.warning(f"Could not get canonical hardware status: {e}")
        
        # Add current metrics if available
        if self.performance_metrics:
            latest_metrics = self.performance_metrics[-1]
            status['current_metrics'] = {
                'cpu_usage': latest_metrics.cpu_usage,
                'memory_usage': latest_metrics.memory_usage,
                'gpu_usage': latest_metrics.gpu_usage,
                'gpu_memory_usage': latest_metrics.gpu_memory_usage,
                'temperature': latest_metrics.temperature
            }
        
        return status
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get hardware optimization recommendations using existing utilities."""
        recommendations = {
            'cpu_optimizations': [],
            'gpu_optimizations': [],
            'memory_optimizations': [],
            'general_recommendations': []
        }
        
        try:
            # Get recommendations from canonical hardware manager
            canonical_recommendations = self.canonical_hardware_manager.get_optimization_recommendations()
            
            # Convert to unified format
            if 'cpu_optimizations' in canonical_recommendations:
                recommendations['cpu_optimizations'] = canonical_recommendations['cpu_optimizations']
            
            if 'gpu_optimizations' in canonical_recommendations:
                recommendations['gpu_optimizations'] = canonical_recommendations['gpu_optimizations']
            
            if 'memory_optimizations' in canonical_recommendations:
                recommendations['memory_optimizations'] = canonical_recommendations['memory_optimizations']
            
            # Add architecture-specific recommendations
            if self.architecture_type == ArchitectureType.TAS:
                recommendations['general_recommendations'].append("Consider disabling GPU for tree-based operations")
            elif self.architecture_type == ArchitectureType.NAS:
                recommendations['general_recommendations'].append("Enable mixed precision training for neural networks")
            
        except Exception as e:
            self.logger.warning(f"Could not get optimization recommendations: {e}")
        
        return recommendations
    
    def export_hardware_data(self, filepath: str):
        """Export hardware performance data to file."""
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'hardware_config': self.config.__dict__,
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
                'recommendations': self.get_optimization_recommendations(),
                'using_existing_utilities': True
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Hardware data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export hardware data: {e}")
            raise


# Convenience functions
def create_hardware_manager(architecture_type: ArchitectureType,
                          optimization_level: CanonicalOptimizationLevel = CanonicalOptimizationLevel.BALANCED,
                          **kwargs) -> UnifiedHardwareManager:
    """Create a hardware manager with default settings using existing utilities."""
    config = HardwareConfig(
        cpu_optimization_level=optimization_level,
        gpu_optimization_level=optimization_level,
        memory_optimization_level=optimization_level,
        **kwargs
    )
    return UnifiedHardwareManager(architecture_type=architecture_type, config=config)


def create_basic_hardware_manager(architecture_type: ArchitectureType) -> UnifiedHardwareManager:
    """Create a basic hardware manager using existing utilities."""
    config = HardwareConfig(
        cpu_optimization_level=CanonicalOptimizationLevel.MINIMAL,
        gpu_optimization_level=CanonicalOptimizationLevel.MINIMAL,
        memory_optimization_level=CanonicalOptimizationLevel.MINIMAL,
        enable_performance_monitoring=False,
        enable_adaptive_optimization=False
    )
    return UnifiedHardwareManager(architecture_type=architecture_type, config=config)


def create_aggressive_hardware_manager(architecture_type: ArchitectureType) -> UnifiedHardwareManager:
    """Create an aggressive hardware manager for maximum performance using existing utilities."""
    config = HardwareConfig(
        cpu_optimization_level=CanonicalOptimizationLevel.MAXIMUM,
        gpu_optimization_level=CanonicalOptimizationLevel.MAXIMUM,
        memory_optimization_level=CanonicalOptimizationLevel.MAXIMUM,
        enable_performance_monitoring=True,
        enable_adaptive_optimization=True,
        monitoring_interval=1.0
    )
    return UnifiedHardwareManager(architecture_type=architecture_type, config=config)