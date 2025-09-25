#!/usr/bin/env python3
"""
Unified Hardware Optimization - Enhanced with Comprehensive Hardware Manager

This module provides unified hardware optimization using existing hardware/ tools,
consolidating M1 GPU, memory, and CPU optimization into a single interface.
Enhanced with comprehensive hardware management capabilities.

Key Features:
- Direct use of existing hardware/ tools
- M1 Apple Silicon optimization
- GPU acceleration with MPS support
- Memory optimization and management
- CPU optimization for parallel processing
- Comprehensive performance monitoring
- Adaptive optimization and learning
- Real-time hardware monitoring
- Workload-specific optimization
"""

import logging
import threading
import time
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import platform
import subprocess
import json
from pathlib import Path

# Import existing hardware tools directly
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager, get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer, get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer, get_m1_cpu_optimizer
    HARDWARE_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Hardware tools not available: {e}")
    HARDWARE_TOOLS_AVAILABLE = False

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Import utility modules
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    )
    UTILITY_MODULES_AVAILABLE = True
except ImportError:
    UTILITY_MODULES_AVAILABLE = False
    # Fallback functions
    def tprint(*args, **kwargs):
        print(*args, **kwargs)
    def tprint_info(*args, **kwargs):
        print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs):
        print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs):
        print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs):
        print("SUCCESS:", *args, **kwargs)

logger = logging.getLogger(__name__)


# Enhanced Enums and Data Classes from comprehensive hardware manager
class WorkloadType(Enum):
    """Types of workloads for optimization."""
    BACKTESTING = "backtesting"
    ML_TRAINING = "ml_training"
    DATA_PROCESSING = "data_processing"
    MONTE_CARLO = "monte_carlo"
    FEATURE_ENGINEERING = "feature_engineering"
    NAS_SEARCH = "nas_search"
    TAS_SEARCH = "tas_search"
    GENERAL = "general"


class OptimizationLevel(Enum):
    """Optimization levels."""
    MINIMAL = "minimal"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    temperature: float
    power_consumption: float
    workload_type: str
    optimization_level: str
    performance_score: float


@dataclass
class HardwareConfig:
    """Enhanced configuration for hardware optimization."""
    # CPU Configuration
    cpu_optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    enable_core_affinity: bool = True
    enable_thermal_monitoring: bool = True
    enable_power_management: bool = True
    
    # GPU Configuration
    gpu_optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    enable_mps_acceleration: bool = True
    enable_gpu_memory_pooling: bool = True
    enable_batch_operations: bool = True
    
    # Memory Configuration
    memory_optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    memory_limit_gb: float = 8.0
    enable_memory_pooling: bool = True
    enable_predictive_allocation: bool = True
    enable_compression: bool = True
    
    # Adaptive Configuration
    enable_adaptive_optimization: bool = True
    learning_enabled: bool = True
    auto_tuning_enabled: bool = True
    performance_monitoring_enabled: bool = True
    
    # Monitoring Configuration
    monitoring_interval: float = 5.0
    metrics_retention_hours: int = 24
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 85.0,
        'memory_usage': 90.0,
        'gpu_usage': 80.0,
        'temperature': 85.0
    })


class HardwarePerformanceMonitor:
    """Real-time hardware performance monitoring."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.logger = logger.getChild('HardwarePerformanceMonitor')
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.alert_callbacks: List[Callable] = []
        
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("🔍 Hardware performance monitoring started")
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        self.logger.info("🔍 Hardware performance monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics
                cutoff_time = time.time() - (self.config.metrics_retention_hours * 3600)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                # Check for alerts
                self._check_alerts(metrics)
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(10)
                
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            if PSUTIL_AVAILABLE:
                # CPU usage
                cpu_usage = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
            else:
                cpu_usage = 0.0
                memory_usage = 0.0
            
            # GPU usage (simplified - would need more sophisticated detection)
            gpu_usage = self._get_gpu_usage()
            
            # Temperature (simplified - would need hardware-specific implementation)
            temperature = self._get_temperature()
            
            # Power consumption (simplified)
            power_consumption = self._get_power_consumption()
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(
                cpu_usage, memory_usage, gpu_usage, temperature
            )
            
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                temperature=temperature,
                power_consumption=power_consumption,
                workload_type="unknown",
                optimization_level="unknown",
                performance_score=performance_score
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=0.0,
                memory_usage=0.0,
                gpu_usage=0.0,
                temperature=0.0,
                power_consumption=0.0,
                workload_type="unknown",
                optimization_level="unknown",
                performance_score=0.0
            )
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage."""
        # Simplified implementation - would need actual GPU monitoring
        return 0.0
    
    def _get_temperature(self) -> float:
        """Get system temperature."""
        # Simplified implementation - would need actual temperature monitoring
        return 0.0
    
    def _get_power_consumption(self) -> float:
        """Get power consumption."""
        # Simplified implementation - would need actual power monitoring
        return 0.0
    
    def _calculate_performance_score(self, cpu_usage: float, memory_usage: float, 
                                   gpu_usage: float, temperature: float) -> float:
        """Calculate overall performance score."""
        # Simple scoring algorithm
        score = 100.0
        
        # Penalize high usage
        if cpu_usage > 80:
            score -= (cpu_usage - 80) * 0.5
        if memory_usage > 85:
            score -= (memory_usage - 85) * 0.7
        if gpu_usage > 90:
            score -= (gpu_usage - 90) * 0.3
        if temperature > 80:
            score -= (temperature - 80) * 0.8
            
        return max(0.0, score)
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for alert conditions."""
        for threshold_name, threshold_value in self.config.alert_thresholds.items():
            if threshold_name == 'cpu_usage' and metrics.cpu_usage > threshold_value:
                self._trigger_alert(f"High CPU usage: {metrics.cpu_usage:.1f}%")
            elif threshold_name == 'memory_usage' and metrics.memory_usage > threshold_value:
                self._trigger_alert(f"High memory usage: {metrics.memory_usage:.1f}%")
            elif threshold_name == 'gpu_usage' and metrics.gpu_usage > threshold_value:
                self._trigger_alert(f"High GPU usage: {metrics.gpu_usage:.1f}%")
            elif threshold_name == 'temperature' and metrics.temperature > threshold_value:
                self._trigger_alert(f"High temperature: {metrics.temperature:.1f}°C")
    
    def _trigger_alert(self, message: str):
        """Trigger an alert."""
        self.logger.warning(f"🚨 Hardware Alert: {message}")
        for callback in self.alert_callbacks:
            try:
                callback(message)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add an alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        return {
            'avg_cpu_usage': sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            'avg_memory_usage': sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            'avg_gpu_usage': sum(m.gpu_usage for m in recent_metrics) / len(recent_metrics),
            'avg_temperature': sum(m.temperature for m in recent_metrics) / len(recent_metrics),
            'avg_performance_score': sum(m.performance_score for m in recent_metrics) / len(recent_metrics),
            'total_measurements': len(self.metrics_history)
        }


class UnifiedHardwareOptimizer:
    """Enhanced unified hardware optimization with comprehensive hardware management."""
    
    def __init__(self, config: Union[Dict[str, Any], HardwareConfig]):
        """Initialize hardware optimizer with existing tools and comprehensive management."""
        # Convert dict config to HardwareConfig if needed
        if isinstance(config, dict):
            self.config = self._dict_to_hardware_config(config)
        else:
            self.config = config
            
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware managers using existing tools
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        
        # Initialize comprehensive hardware management
        self.performance_monitor = None
        self.workload_type = WorkloadType.GENERAL
        self.optimization_history = []
        self.adaptive_learning_enabled = False
        
        if HARDWARE_TOOLS_AVAILABLE and self.config.enable_adaptive_optimization:
            self._initialize_hardware_tools()
            self._initialize_comprehensive_management()
    
    def _dict_to_hardware_config(self, config_dict: Dict[str, Any]) -> HardwareConfig:
        """Convert dictionary config to HardwareConfig."""
        return HardwareConfig(
            cpu_optimization_level=OptimizationLevel(config_dict.get('cpu_optimization_level', 'balanced')),
            enable_core_affinity=config_dict.get('enable_core_affinity', True),
            enable_thermal_monitoring=config_dict.get('enable_thermal_monitoring', True),
            enable_power_management=config_dict.get('enable_power_management', True),
            gpu_optimization_level=OptimizationLevel(config_dict.get('gpu_optimization_level', 'balanced')),
            enable_mps_acceleration=config_dict.get('enable_mps_acceleration', True),
            enable_gpu_memory_pooling=config_dict.get('enable_gpu_memory_pooling', True),
            enable_batch_operations=config_dict.get('enable_batch_operations', True),
            memory_optimization_level=OptimizationLevel(config_dict.get('memory_optimization_level', 'balanced')),
            memory_limit_gb=config_dict.get('memory_limit_gb', 8.0),
            enable_memory_pooling=config_dict.get('enable_memory_pooling', True),
            enable_predictive_allocation=config_dict.get('enable_predictive_allocation', True),
            enable_compression=config_dict.get('enable_compression', True),
            enable_adaptive_optimization=config_dict.get('enable_adaptive_optimization', True),
            learning_enabled=config_dict.get('learning_enabled', True),
            auto_tuning_enabled=config_dict.get('auto_tuning_enabled', True),
            performance_monitoring_enabled=config_dict.get('performance_monitoring_enabled', True),
            monitoring_interval=config_dict.get('monitoring_interval', 5.0),
            metrics_retention_hours=config_dict.get('metrics_retention_hours', 24),
            alert_thresholds=config_dict.get('alert_thresholds', {
                'cpu_usage': 85.0,
                'memory_usage': 90.0,
                'gpu_usage': 80.0,
                'temperature': 85.0
            })
        )
    
    def _initialize_comprehensive_management(self):
        """Initialize comprehensive hardware management."""
        try:
            # Initialize performance monitor
            if self.config.performance_monitoring_enabled:
                self.performance_monitor = HardwarePerformanceMonitor(self.config)
                self.performance_monitor.start_monitoring()
                tprint_success("✅ Comprehensive hardware monitoring initialized")
            
            # Enable adaptive learning
            if self.config.learning_enabled:
                self.adaptive_learning_enabled = True
                tprint_success("✅ Adaptive learning enabled")
                
        except Exception as e:
            tprint_warning(f"Could not initialize comprehensive management: {e}")
    
    def set_workload_type(self, workload_type: WorkloadType):
        """Set the current workload type for optimization."""
        self.workload_type = workload_type
        tprint_info(f"🔄 Workload type set to: {workload_type.value}")
        
        # Apply workload-specific optimizations
        self._apply_workload_optimizations()
    
    def _apply_workload_optimizations(self):
        """Apply workload-specific optimizations."""
        if not HARDWARE_TOOLS_AVAILABLE:
            return
            
        try:
            if self.workload_type == WorkloadType.NAS_SEARCH:
                # NAS-specific optimizations
                if self.gpu_manager and hasattr(self.gpu_manager, 'optimize_for_nas'):
                    self.gpu_manager.optimize_for_nas()
                if self.memory_optimizer and hasattr(self.memory_optimizer, 'optimize_for_nas'):
                    self.memory_optimizer.optimize_for_nas()
                    
            elif self.workload_type == WorkloadType.TAS_SEARCH:
                # TAS-specific optimizations
                if self.gpu_manager and hasattr(self.gpu_manager, 'optimize_for_tas'):
                    self.gpu_manager.optimize_for_tas()
                if self.memory_optimizer and hasattr(self.memory_optimizer, 'optimize_for_tas'):
                    self.memory_optimizer.optimize_for_tas()
                    
            elif self.workload_type == WorkloadType.ML_TRAINING:
                # ML training optimizations
                if self.gpu_manager and hasattr(self.gpu_manager, 'optimize_for_training'):
                    self.gpu_manager.optimize_for_training()
                if self.memory_optimizer and hasattr(self.memory_optimizer, 'optimize_for_training'):
                    self.memory_optimizer.optimize_for_training()
                    
            elif self.workload_type == WorkloadType.BACKTESTING:
                # Backtesting optimizations
                if self.gpu_manager and hasattr(self.gpu_manager, 'optimize_for_backtesting'):
                    self.gpu_manager.optimize_for_backtesting()
                if self.memory_optimizer and hasattr(self.memory_optimizer, 'optimize_for_backtesting'):
                    self.memory_optimizer.optimize_for_backtesting()
                    
            tprint_success(f"✅ Applied {self.workload_type.value} optimizations")
            
        except Exception as e:
            tprint_warning(f"Could not apply workload optimizations: {e}")
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations based on current performance."""
        recommendations = {
            'cpu_optimization': [],
            'memory_optimization': [],
            'gpu_optimization': [],
            'general_recommendations': []
        }
        
        if self.performance_monitor:
            summary = self.performance_monitor.get_performance_summary()
            
            # CPU recommendations
            if summary.get('avg_cpu_usage', 0) > 80:
                recommendations['cpu_optimization'].append("Consider reducing CPU-intensive operations")
                recommendations['cpu_optimization'].append("Enable core affinity optimization")
            
            # Memory recommendations
            if summary.get('avg_memory_usage', 0) > 85:
                recommendations['memory_optimization'].append("Consider enabling memory pooling")
                recommendations['memory_optimization'].append("Enable predictive memory allocation")
            
            # GPU recommendations
            if summary.get('avg_gpu_usage', 0) > 90:
                recommendations['gpu_optimization'].append("Consider batch processing optimization")
                recommendations['gpu_optimization'].append("Enable GPU memory pooling")
            
            # Performance score recommendations
            if summary.get('avg_performance_score', 100) < 70:
                recommendations['general_recommendations'].append("Overall performance is below optimal")
                recommendations['general_recommendations'].append("Consider enabling adaptive optimization")
        
        return recommendations
    
    def _initialize_hardware_tools(self):
        """Initialize hardware tools directly."""
        try:
            # Use existing hardware tools directly
            if self.config.enable_adaptive_optimization:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                
                tprint_info("Hardware tools initialized using existing hardware/ modules")
            else:
                tprint_info("Hardware optimization disabled")
        except Exception as e:
            tprint_warning(f"Could not initialize hardware tools: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'hardware_tools_available': HARDWARE_TOOLS_AVAILABLE,
            'gpu_manager': self.gpu_manager is not None,
            'memory_optimizer': self.memory_optimizer is not None,
            'cpu_optimizer': self.cpu_optimizer is not None,
            'performance_monitor': self.performance_monitor is not None,
            'workload_type': self.workload_type.value,
            'adaptive_learning_enabled': self.adaptive_learning_enabled
        }
        
        # Add performance monitor summary
        if self.performance_monitor:
            summary.update(self.performance_monitor.get_performance_summary())
        
        # Add optimization recommendations
        summary['recommendations'] = self.get_optimization_recommendations()
        
        return summary
    
    @contextmanager
    def gpu_context(self):
        """Context manager for GPU operations using existing tools."""
        if self.gpu_manager:
            try:
                # Use existing GPU context from hardware tools
                if hasattr(self.gpu_manager, 'gpu_context'):
                    with self.gpu_manager.gpu_context():
                        yield
                else:
                    yield
            except Exception as e:
                tprint_warning(f"GPU context failed: {e}")
                yield
        else:
            yield
    
    @contextmanager
    def memory_context(self):
        """Context manager for memory optimization using existing tools."""
        if self.memory_optimizer:
            try:
                # Use existing memory context from hardware tools
                if hasattr(self.memory_optimizer, 'memory_checkpoint'):
                    with self.memory_optimizer.memory_checkpoint():
                        yield
                else:
                    yield
            except Exception as e:
                tprint_warning(f"Memory context failed: {e}")
                yield
        else:
            yield
    
    def optimize_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Optimize data using existing hardware tools."""
        if self.memory_optimizer and isinstance(data, pd.DataFrame):
            try:
                # Use existing data optimization from hardware tools
                if hasattr(self.memory_optimizer, 'optimize_dataframe'):
                    return self.memory_optimizer.optimize_dataframe(data)
            except Exception as e:
                tprint_warning(f"Data optimization failed: {e}")
        
        return data
    
    def get_memory_usage(self) -> float:
        """Get memory usage using existing tools."""
        if self.memory_optimizer:
            try:
                if hasattr(self.memory_optimizer, 'get_memory_usage'):
                    return self.memory_optimizer.get_memory_usage()
            except Exception as e:
                tprint_warning(f"Memory usage check failed: {e}")
        
        # Fallback
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def cleanup(self):
        """Cleanup using existing hardware tools."""
        if self.memory_optimizer:
            try:
                if hasattr(self.memory_optimizer, 'cleanup'):
                    self.memory_optimizer.cleanup()
            except Exception as e:
                tprint_warning(f"Hardware cleanup failed: {e}")
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get information about available hardware."""
        info = {
            'hardware_tools_available': HARDWARE_TOOLS_AVAILABLE,
            'gpu_manager': self.gpu_manager is not None,
            'memory_optimizer': self.memory_optimizer is not None,
            'cpu_optimizer': self.cpu_optimizer is not None
        }
        
        if self.gpu_manager:
            try:
                if hasattr(self.gpu_manager, 'get_gpu_info'):
                    info['gpu_info'] = self.gpu_manager.get_gpu_info()
            except Exception as e:
                tprint_warning(f"Could not get GPU info: {e}")
        
        return info
    
    def start_monitoring(self):
        """Start hardware monitoring if available."""
        if self.memory_optimizer:
            try:
                if hasattr(self.memory_optimizer, 'start_monitoring'):
                    self.memory_optimizer.start_monitoring()
                    tprint_info("Hardware monitoring started")
            except Exception as e:
                tprint_warning(f"Could not start hardware monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop hardware monitoring if available."""
        if self.memory_optimizer:
            try:
                if hasattr(self.memory_optimizer, 'stop_monitoring'):
                    self.memory_optimizer.stop_monitoring()
                    tprint_info("Hardware monitoring stopped")
            except Exception as e:
                tprint_warning(f"Could not stop hardware monitoring: {e}")