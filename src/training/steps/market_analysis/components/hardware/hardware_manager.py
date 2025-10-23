"""
Hardware Manager for Market Analysis Components.

This module provides comprehensive hardware management capabilities for
market analysis pipeline steps, including device selection, resource
monitoring, and optimization coordination.
"""

import platform
import psutil
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import threading
import time

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error
from src.utils.common_operations import (
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    is_m1_available, is_mps_available
)
from src.utils.hardware.m1_gpu_utils import M1GPUManager
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.training.steps.market_analysis.components.base_component import BaseMarketAnalysisComponent, ComponentConfig

class DeviceType(Enum):
    """Available device types."""
    CPU = "cpu"
    GPU = "gpu"
    M1_GPU = "m1_gpu"
    M1_CPU = "m1_cpu"
    AUTO = "auto"

class OptimizationLevel(Enum):
    """Hardware optimization levels."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

@dataclass
class HardwareCapabilities:
    """Hardware capabilities and configuration."""
    has_gpu: bool = False
    gpu_memory_gb: float = 0.0
    has_m1: bool = False
    m1_memory_gb: float = 0.0
    cpu_cores: int = 1
    total_memory_gb: float = 0.0
    optimal_batch_size: int = 1000
    recommended_workers: int = 1
    device_type: DeviceType = DeviceType.CPU
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC

@dataclass
class HardwareConfig:
    """Configuration for hardware management."""
    # Device selection
    preferred_device: DeviceType = DeviceType.AUTO
    fallback_device: DeviceType = DeviceType.CPU
    
    # Memory management
    max_memory_usage: float = 0.8  # 80% of available memory
    memory_cleanup_threshold: float = 0.9
    enable_memory_monitoring: bool = True
    
    # Performance optimization
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None
    
    # M1 specific
    enable_m1_optimization: bool = True
    m1_memory_pressure_threshold: float = 0.7
    
    # GPU specific
    enable_gpu_acceleration: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval: float = 5.0  # seconds

@dataclass
class HardwareStatus:
    """Current hardware status."""
    device_type: DeviceType
    memory_usage: float
    cpu_usage: float
    gpu_usage: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    optimization_active: bool = False
    last_updated: datetime = field(default_factory=datetime.now)

class HardwareManager(BaseMarketAnalysisComponent):
    """
    Comprehensive hardware manager for market analysis components.
    
    Provides:
    - Device detection and selection
    - Resource monitoring and management
    - Optimization coordination
    - Performance tracking
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        """Initialize the hardware manager."""
        super().__init__(ComponentConfig())
        self.hardware_config = config or HardwareConfig()
        self.logger = logging.getLogger(__name__)
        
        # Hardware capabilities
        self.capabilities = self._detect_hardware_capabilities()
        
        # Current status
        self.status = HardwareStatus(
            device_type=self.capabilities.device_type,
            memory_usage=0.0,
            cpu_usage=0.0,
            optimization_active=False
        )
        
        # Optimization components
        self.gpu_manager = None
        self.memory_optimizer = None
        self.m1_optimizer = None
        self.performance_monitor = None
        
        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Initialize hardware components
        self._initialize_hardware_components()
        
    def _detect_hardware_capabilities(self) -> HardwareCapabilities:
        """Detect available hardware capabilities."""
        try:
            # Basic system info
            cpu_cores = psutil.cpu_count(logical=True)
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # GPU detection
            has_gpu = False
            gpu_memory_gb = 0.0
            
            try:
                import torch
                if torch.cuda.is_available():
                    has_gpu = True
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except ImportError:
                pass
            
            # M1 detection
            has_m1 = is_m1_available()
            m1_memory_gb = 0.0
            if has_m1:
                m1_memory_gb = total_memory_gb  # M1 systems use unified memory
            
            # Determine optimal device
            device_type = DeviceType.CPU
            if has_m1 and self.hardware_config.enable_m1_optimization:
                device_type = DeviceType.M1_CPU
            elif has_gpu and self.hardware_config.enable_gpu_acceleration:
                device_type = DeviceType.GPU
            
            # Calculate optimal batch size
            if device_type in [DeviceType.GPU, DeviceType.M1_GPU]:
                optimal_batch_size = min(10000, int(total_memory_gb * 1000))
            else:
                optimal_batch_size = min(5000, int(total_memory_gb * 500))
            
            # Calculate recommended workers
            recommended_workers = min(cpu_cores, 8) if device_type == DeviceType.CPU else 1
            
            return HardwareCapabilities(
                has_gpu=has_gpu,
                gpu_memory_gb=gpu_memory_gb,
                has_m1=has_m1,
                m1_memory_gb=m1_memory_gb,
                cpu_cores=cpu_cores,
                total_memory_gb=total_memory_gb,
                optimal_batch_size=optimal_batch_size,
                recommended_workers=recommended_workers,
                device_type=device_type,
                optimization_level=OptimizationLevel.BASIC
            )
            
        except Exception as e:
            tprint_warning(f"Hardware detection failed: {str(e)}")
            return HardwareCapabilities()
    
    def _initialize_hardware_components(self):
        """Initialize hardware optimization components."""
        try:
            # Initialize GPU manager if available
            if self.capabilities.has_gpu and self.hardware_config.enable_gpu_acceleration:
                try:
                    from .gpu_accelerator import GPUAccelerator, GPUConfig
                    gpu_config = GPUConfig(
                        memory_fraction=self.hardware_config.gpu_memory_fraction,
                        enable_mixed_precision=True
                    )
                    self.gpu_manager = GPUAccelerator(gpu_config)
                    tprint_info("✅ GPU accelerator initialized")
                except Exception as e:
                    tprint_warning(f"GPU accelerator initialization failed: {str(e)}")
            
            # Initialize memory optimizer
            try:
                from .memory_optimizer import MemoryOptimizer, MemoryConfig
                memory_config = MemoryConfig(
                    max_usage=self.hardware_config.max_memory_usage,
                    cleanup_threshold=self.hardware_config.memory_cleanup_threshold
                )
                self.memory_optimizer = MemoryOptimizer(memory_config)
                tprint_info("✅ Memory optimizer initialized")
            except Exception as e:
                tprint_warning(f"Memory optimizer initialization failed: {str(e)}")
            
            # Initialize M1 optimizer if available
            if self.capabilities.has_m1 and self.hardware_config.enable_m1_optimization:
                try:
                    from .m1_optimizer import M1Optimizer, M1Config
                    m1_config = M1Config(
                        memory_pressure_threshold=self.hardware_config.m1_memory_pressure_threshold,
                        enable_vectorization=True
                    )
                    self.m1_optimizer = M1Optimizer(m1_config)
                    tprint_info("✅ M1 optimizer initialized")
                except Exception as e:
                    tprint_warning(f"M1 optimizer initialization failed: {str(e)}")
            
            # Initialize performance monitor
            if self.hardware_config.enable_performance_monitoring:
                try:
                    from .performance_monitor import PerformanceMonitor, PerformanceConfig
                    perf_config = PerformanceConfig(
                        monitoring_interval=self.hardware_config.monitoring_interval,
                        enable_gpu_monitoring=self.capabilities.has_gpu
                    )
                    self.performance_monitor = PerformanceMonitor(perf_config)
                    tprint_info("✅ Performance monitor initialized")
                except Exception as e:
                    tprint_warning(f"Performance monitor initialization failed: {str(e)}")
            
        except Exception as e:
            tprint_error(f"Hardware components initialization failed: {str(e)}")
    
    async def optimize_for_task(self, 
                              task_type: str,
                              data_size: int,
                              complexity: str = "medium") -> Dict[str, Any]:
        """
        Optimize hardware configuration for a specific task.
        
        Args:
            task_type: Type of task (clustering, training, inference, etc.)
            data_size: Size of data to process
            complexity: Task complexity (low, medium, high)
            
        Returns:
            Dictionary with optimization recommendations
        """
        try:
            tprint_info(f"🔧 Optimizing hardware for {task_type} task (size: {data_size}, complexity: {complexity})")
            
            # Determine optimal device
            optimal_device = self._select_optimal_device(task_type, data_size, complexity)
            
            # Calculate optimal batch size
            optimal_batch_size = self._calculate_optimal_batch_size(data_size, complexity)
            
            # Determine memory requirements
            memory_requirements = self._calculate_memory_requirements(data_size, complexity)
            
            # Generate optimization recommendations
            recommendations = {
                'device_type': optimal_device.value,
                'batch_size': optimal_batch_size,
                'memory_requirements_gb': memory_requirements,
                'parallel_workers': self._calculate_optimal_workers(task_type, complexity),
                'optimization_level': self._get_optimization_level(complexity).value,
                'memory_cleanup': memory_requirements > self.capabilities.total_memory_gb * 0.5,
                'enable_mixed_precision': complexity in ['high', 'maximum'],
                'enable_gradient_checkpointing': complexity == 'maximum'
            }
            
            # Apply optimizations
            await self._apply_optimizations(recommendations)
            
            tprint_info(f"✅ Hardware optimization completed: {optimal_device.value}")
            return recommendations
            
        except Exception as e:
            tprint_error(f"❌ Hardware optimization failed: {str(e)}")
            return {'error': str(e)}
    
    def _select_optimal_device(self, task_type: str, data_size: int, complexity: str) -> DeviceType:
        """Select optimal device for the task."""
        # GPU is preferred for large data and complex tasks
        if (self.capabilities.has_gpu and 
            data_size > 10000 and 
            complexity in ['high', 'maximum'] and
            self.hardware_config.enable_gpu_acceleration):
            return DeviceType.GPU
        
        # M1 optimization for medium tasks
        if (self.capabilities.has_m1 and 
            data_size > 1000 and 
            self.hardware_config.enable_m1_optimization):
            return DeviceType.M1_CPU
        
        # CPU for small tasks or when others are not available
        return DeviceType.CPU
    
    def _calculate_optimal_batch_size(self, data_size: int, complexity: str) -> int:
        """Calculate optimal batch size for the task."""
        base_batch_size = self.capabilities.optimal_batch_size
        
        # Adjust based on complexity
        complexity_multipliers = {
            'low': 1.5,
            'medium': 1.0,
            'high': 0.7,
            'maximum': 0.5
        }
        
        multiplier = complexity_multipliers.get(complexity, 1.0)
        optimal_batch_size = int(base_batch_size * multiplier)
        
        # Ensure it doesn't exceed data size
        return min(optimal_batch_size, data_size)
    
    def _calculate_memory_requirements(self, data_size: int, complexity: str) -> float:
        """Calculate estimated memory requirements in GB."""
        # Base memory per data point (rough estimate)
        base_memory_per_point = 0.001  # 1MB per 1000 points
        
        # Complexity multipliers
        complexity_multipliers = {
            'low': 1.0,
            'medium': 2.0,
            'high': 4.0,
            'maximum': 8.0
        }
        
        multiplier = complexity_multipliers.get(complexity, 1.0)
        memory_gb = (data_size * base_memory_per_point * multiplier) / 1000
        
        return min(memory_gb, self.capabilities.total_memory_gb * 0.8)
    
    def _calculate_optimal_workers(self, task_type: str, complexity: str) -> int:
        """Calculate optimal number of workers."""
        if not self.hardware_config.enable_parallel_processing:
            return 1
        
        # Task-specific worker recommendations
        if task_type in ['clustering', 'training']:
            return min(self.capabilities.recommended_workers, 4)
        elif task_type in ['inference', 'prediction']:
            return min(self.capabilities.recommended_workers, 2)
        else:
            return 1
    
    def _get_optimization_level(self, complexity: str) -> OptimizationLevel:
        """Get optimization level based on complexity."""
        complexity_levels = {
            'low': OptimizationLevel.BASIC,
            'medium': OptimizationLevel.BASIC,
            'high': OptimizationLevel.AGGRESSIVE,
            'maximum': OptimizationLevel.MAXIMUM
        }
        
        return complexity_levels.get(complexity, OptimizationLevel.BASIC)
    
    async def _apply_optimizations(self, recommendations: Dict[str, Any]):
        """Apply hardware optimizations based on recommendations."""
        try:
            # Update device type
            device_type = DeviceType(recommendations['device_type'])
            self.status.device_type = device_type
            
            # Apply memory optimizations
            if self.memory_optimizer and recommendations.get('memory_cleanup', False):
                await self.memory_optimizer.cleanup_memory()
            
            # Apply M1 optimizations
            if self.m1_optimizer and device_type in [DeviceType.M1_CPU, DeviceType.M1_GPU]:
                await self.m1_optimizer.optimize_for_task(recommendations)
            
            # Apply GPU optimizations
            if self.gpu_manager and device_type == DeviceType.GPU:
                await self.gpu_manager.optimize_for_task(recommendations)
            
            # Start performance monitoring
            if self.performance_monitor and not self.monitoring_active:
                await self.start_monitoring()
            
            self.status.optimization_active = True
            
        except Exception as e:
            tprint_warning(f"Optimization application failed: {str(e)}")
    
    async def start_monitoring(self):
        """Start hardware performance monitoring."""
        if self.performance_monitor and not self.monitoring_active:
            await self.performance_monitor.start_monitoring()
            self.monitoring_active = True
            tprint_info("🔍 Hardware monitoring started")
    
    async def stop_monitoring(self):
        """Stop hardware performance monitoring."""
        if self.performance_monitor and self.monitoring_active:
            await self.performance_monitor.stop_monitoring()
            self.monitoring_active = False
            tprint_info("⏹️ Hardware monitoring stopped")
    
    async def get_hardware_status(self) -> HardwareStatus:
        """Get current hardware status."""
        try:
            # Update memory usage
            memory_info = psutil.virtual_memory()
            self.status.memory_usage = memory_info.percent / 100.0
            
            # Update CPU usage
            self.status.cpu_usage = psutil.cpu_percent() / 100.0
            
            # Update GPU usage if available
            if self.gpu_manager:
                gpu_status = await self.gpu_manager.get_status()
                self.status.gpu_usage = gpu_status.get('gpu_usage')
                self.status.gpu_memory_usage = gpu_status.get('gpu_memory_usage')
            
            self.status.last_updated = datetime.now()
            return self.status
            
        except Exception as e:
            tprint_warning(f"Status update failed: {str(e)}")
            return self.status
    
    async def cleanup_resources(self):
        """Cleanup hardware resources."""
        try:
            # Stop monitoring
            await self.stop_monitoring()
            
            # Cleanup memory
            if self.memory_optimizer:
                await self.memory_optimizer.cleanup_memory()
            
            # Cleanup GPU resources
            if self.gpu_manager:
                await self.gpu_manager.cleanup()
            
            # Cleanup M1 resources
            if self.m1_optimizer:
                await self.m1_optimizer.cleanup()
            
            self.status.optimization_active = False
            tprint_info("🧹 Hardware resources cleaned up")
            
        except Exception as e:
            tprint_warning(f"Resource cleanup failed: {str(e)}")
    
    def get_capabilities(self) -> HardwareCapabilities:
        """Get hardware capabilities."""
        return self.capabilities
    
    def get_config(self) -> HardwareConfig:
        """Get hardware configuration."""
        return self.hardware_config
    
    def update_config(self, new_config: HardwareConfig):
        """Update hardware configuration."""
        self.hardware_config = new_config
        # Reinitialize components if needed
        self._initialize_hardware_components()
        tprint_info("🔧 Hardware configuration updated")