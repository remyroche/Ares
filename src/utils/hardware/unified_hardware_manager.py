"""
Unified Hardware Manager for Apple Silicon.

This module provides a centralized hardware management system that coordinates
all hardware optimizations including CPU, GPU, memory, and adaptive optimization.
"""

import logging
import threading
import time
import asyncio
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import platform
import subprocess
from contextlib import contextmanager
import json
from pathlib import Path

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from .m1_cpu_optimizer import M1CPUOptimizer
from .m1_gpu_utils import M1GPUManager
from .m1_memory_optimizer import M1MemoryOptimizer

logger = logging.getLogger(__name__)

class WorkloadType(Enum):
    """Types of workloads for optimization."""
    BACKTESTING = "backtesting"
    ML_TRAINING = "ml_training"
    DATA_PROCESSING = "data_processing"
    MONTE_CARLO = "monte_carlo"
    FEATURE_ENGINEERING = "feature_engineering"
    GENERAL = "general"

class OptimizationLevel(Enum):
    """Optimization levels."""
    MINIMAL = "minimal"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

@dataclass
class HardwareConfig:
    """Configuration for hardware optimization."""
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
    enable_learning: bool = True  # Rename learning_enabled to enable_learning
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

    # Intensive workload thresholds for operations like clustering and feature generation
    intensive_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 95.0,  # Allow higher CPU usage for intensive operations
        'memory_usage': 95.0,
        'gpu_usage': 85.0,
        'temperature': 90.0
    })

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
        try:
            # This is a simplified implementation
            # In practice, you'd use platform-specific tools
            if platform.system() == 'Darwin':
                # Try to get GPU usage from system
                result = subprocess.run(
                    ['system_profiler', 'SPDisplaysDataType'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Parse output to get GPU usage (simplified)
                    return 0.0  # Placeholder
            return 0.0
        except Exception:
            return 0.0

    def _get_temperature(self) -> float:
        """Get system temperature."""
        try:
            # Disable system temperature monitoring to avoid sudo requirements
            # Return a default temperature to prevent system calls
            self.logger.debug("Temperature monitoring disabled to avoid sudo requirements")
            return 45.0
        except Exception:
            return 45.0

    def _get_power_consumption(self) -> float:
        """Get power consumption in watts."""
        try:
            # Simplified implementation
            return 15.0  # Placeholder
        except Exception:
            return 15.0

    def _calculate_performance_score(self, cpu: float, memory: float, gpu: float, temp: float) -> float:
        """Calculate overall performance score."""
        # Normalize metrics (lower is better for some)
        cpu_score = max(0, 100 - cpu)
        memory_score = max(0, 100 - memory)
        gpu_score = max(0, 100 - gpu)
        temp_score = max(0, 100 - (temp - 30) * 2)  # Optimal around 30-40°C

        # Weighted average
        return (cpu_score * 0.3 + memory_score * 0.3 + gpu_score * 0.2 + temp_score * 0.2)

    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for alert conditions."""
        for metric_name, threshold in self.config.alert_thresholds.items():
            value = getattr(metrics, metric_name, 0)
            if value > threshold:
                self._trigger_alert(metric_name, value, threshold)

    def _trigger_alert(self, metric_name: str, value: float, threshold: float):
        """Trigger alert for metric threshold breach."""
        alert_msg = f"🚨 ALERT: {metric_name} ({value:.1f}) exceeds threshold ({threshold:.1f})"
        self.logger.warning(alert_msg)

        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(metric_name, value, threshold)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")

    def add_alert_callback(self, callback: Callable):
        """Add alert callback function."""

    def set_intensive_thresholds(self):
        """Switch to intensive workload thresholds for CPU/GPU intensive operations."""
        self.config.alert_thresholds = self.config.intensive_thresholds.copy()
        self.logger.info("🔧 Switched to intensive workload thresholds")

    def set_normal_thresholds(self):
        """Switch back to normal thresholds."""
        self.config.alert_thresholds = {
            'cpu_usage': 85.0,
            'memory_usage': 90.0,
            'gpu_usage': 80.0,
            'temperature': 85.0
        }
        self.logger.info("🔧 Switched back to normal thresholds")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        if not self.metrics_history:
            return {"error": "No metrics available"}

        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements

        return {
            "current_metrics": recent_metrics[-1].__dict__ if recent_metrics else {},
            "average_metrics": {
                "cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
                "memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
                "gpu_usage": sum(m.gpu_usage for m in recent_metrics) / len(recent_metrics),
                "temperature": sum(m.temperature for m in recent_metrics) / len(recent_metrics),
                "performance_score": sum(m.performance_score for m in recent_metrics) / len(recent_metrics)
            },
            "peak_metrics": {
                "max_cpu_usage": max(m.cpu_usage for m in recent_metrics),
                "max_memory_usage": max(m.memory_usage for m in recent_metrics),
                "max_temperature": max(m.temperature for m in recent_metrics),
                "min_performance_score": min(m.performance_score for m in recent_metrics)
            },
            "total_measurements": len(self.metrics_history),
            "monitoring_duration_hours": (time.time() - self.metrics_history[0].timestamp) / 3600 if self.metrics_history else 0
        }

class AdaptiveTaskScheduler:
    """Adaptive task scheduling based on hardware conditions."""

    def __init__(self, config: HardwareConfig):
        self.config = config
        self.logger = logger.getChild('AdaptiveTaskScheduler')
        self.task_queue: List[Dict[str, Any]] = []
        self.running_tasks: Dict[str, Any] = {}
        self.performance_history: List[Dict[str, Any]] = []

    def schedule_task(self, task_id: str, task_func: Callable,
                     workload_type: WorkloadType, priority: int = 5) -> bool:
        """Schedule a task for execution."""
        task = {
            'id': task_id,
            'func': task_func,
            'workload_type': workload_type,
            'priority': priority,
            'created_at': time.time(),
            'status': 'queued'
        }

        self.task_queue.append(task)
        self.task_queue.sort(key=lambda x: x['priority'], reverse=True)

        self.logger.info(f"📋 Task {task_id} scheduled with priority {priority}")
        return True

    def execute_next_task(self, current_metrics: PerformanceMetrics) -> Optional[Any]:
        """Execute the next task in queue based on current conditions."""
        if not self.task_queue:
            return None

        # Select task based on current hardware conditions
        selected_task = self._select_optimal_task(current_metrics)
        if not selected_task:
            return None

        # Remove from queue and add to running
        self.task_queue.remove(selected_task)
        self.running_tasks[selected_task['id']] = selected_task
        selected_task['status'] = 'running'
        selected_task['started_at'] = time.time()

        try:
            self.logger.info(f"🚀 Executing task {selected_task['id']}")
            result = selected_task['func']()

            # Record performance
            execution_time = time.time() - selected_task['started_at']
            self._record_task_performance(selected_task, execution_time, True)

            return result

        except Exception as e:
            self.logger.error(f"❌ Task {selected_task['id']} failed: {e}")
            self._record_task_performance(selected_task, 0, False)
            raise

        finally:
            # Remove from running tasks
            if selected_task['id'] in self.running_tasks:
                del self.running_tasks[selected_task['id']]

    def _select_optimal_task(self, metrics: PerformanceMetrics) -> Optional[Dict[str, Any]]:
        """Select optimal task based on current hardware conditions."""
        if not self.task_queue:
            return None

        # Simple selection logic - can be enhanced with ML
        if metrics.cpu_usage > 80:
            # High CPU usage - prefer lighter tasks
            for task in self.task_queue:
                if task['workload_type'] in [WorkloadType.DATA_PROCESSING, WorkloadType.GENERAL]:
                    return task
        elif metrics.memory_usage > 85:
            # High memory usage - prefer memory-efficient tasks
            for task in self.task_queue:
                if task['workload_type'] in [WorkloadType.BACKTESTING, WorkloadType.MONTE_CARLO]:
                    return task
        else:
            # Normal conditions - use priority
            return self.task_queue[0]

        return self.task_queue[0] if self.task_queue else None

    def _record_task_performance(self, task: Dict[str, Any], execution_time: float, success: bool):
        """Record task performance for learning."""
        performance_record = {
            'task_id': task['id'],
            'workload_type': task['workload_type'].value,
            'execution_time': execution_time,
            'success': success,
            'timestamp': time.time()
        }

        self.performance_history.append(performance_record)

        # Keep only recent history
        cutoff_time = time.time() - (24 * 3600)  # 24 hours
        self.performance_history = [
            r for r in self.performance_history
            if r['timestamp'] > cutoff_time
        ]

    def get_scheduling_report(self) -> Dict[str, Any]:
        """Get scheduling report."""
        return {
            'queued_tasks': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.performance_history),
            'average_execution_time': sum(r['execution_time'] for r in self.performance_history) / max(1, len(self.performance_history)),
            'success_rate': sum(1 for r in self.performance_history if r['success']) / max(1, len(self.performance_history))
        }

class UnifiedHardwareManager:
    """Unified hardware management system."""

    _instance = None
    _init_done = False

    def __init__(self, config: Optional[HardwareConfig] = None):
        # Prevent multiple initialization
        if hasattr(self, '_initialized'):
            return
            
        self.config = config or HardwareConfig()
        self.logger = logger.getChild('UnifiedHardwareManager')

        # Initialize hardware components
        self.cpu_optimizer = M1CPUOptimizer()
        self.gpu_manager = M1GPUManager()
        self.memory_optimizer = M1MemoryOptimizer(
            memory_limit_gb=self.config.memory_limit_gb
        )

        # Initialize management components
        self.performance_monitor = HardwarePerformanceMonitor(self.config)
        self.task_scheduler = AdaptiveTaskScheduler(self.config)

        # State tracking
        self.is_initialized = False
        self.current_workload_type: Optional[WorkloadType] = None
        self.optimization_contexts: Dict[str, Any] = {}
        self._initialized = True
        
        # Circuit breaker for optimization failures
        self._optimization_failures = 0
        self._max_optimization_failures = 3
        self._circuit_breaker_reset_time = None

        self.logger.info("🔧 Unified Hardware Manager initialized")

    @classmethod
    def get_instance(cls, config: Optional[HardwareConfig] = None):
        """Get singleton instance of UnifiedHardwareManager."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def initialize(self) -> bool:
        """Initialize all hardware components."""
        try:
            # Start performance monitoring
            if self.config.performance_monitoring_enabled:
                self.performance_monitor.start_monitoring()

            # Initialize memory optimizer
            self.memory_optimizer.start_monitoring()

            # Set up alert callbacks
            self.performance_monitor.add_alert_callback(self._handle_performance_alert)

            self.is_initialized = True
            self.logger.info("✅ Unified Hardware Manager fully initialized")

            # Mark as initialized to prevent re-initialization
            UnifiedHardwareManager._init_done = True
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Unified Hardware Manager: {e}")
            return False

    def set_intensive_thresholds(self):
        """Switch to intensive workload thresholds for CPU/GPU intensive operations."""
        self.performance_monitor.set_intensive_thresholds()
        self.logger.info("🔧 Switched to intensive workload thresholds")

    def set_normal_thresholds(self):
        """Switch back to normal thresholds."""
        self.performance_monitor.set_normal_thresholds()
        self.logger.info("🔧 Switched back to normal thresholds")

    def configure_workload(self, workload_type, optimization_level):
        """Configure the hardware manager for a specific workload type and optimization level."""
        self.current_workload_type = workload_type
        self.logger.info(f"🔧 Configured for workload: {workload_type}, optimization: {optimization_level}")
        
        # Set thresholds based on optimization level
        if optimization_level.name.upper() == 'INTENSIVE':
            self.set_intensive_thresholds()
        else:
            self.set_normal_thresholds()

    def shutdown(self):
        """Shutdown all components."""
        try:
            self.performance_monitor.stop_monitoring()
            self.memory_optimizer.stop_monitoring()
            self.is_initialized = False
            self.logger.info("🛑 Unified Hardware Manager shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def optimize_for_workload(self, workload_type: WorkloadType,
                            optimization_level: OptimizationLevel = None) -> bool:
        """Optimize hardware for specific workload type."""
        if not self.is_initialized:
            self.logger.warning("Hardware manager not initialized")
            return False

        # Circuit breaker check
        current_time = time.time()
        if self._optimization_failures >= self._max_optimization_failures:
            if self._circuit_breaker_reset_time is None:
                self._circuit_breaker_reset_time = current_time + 300  # 5 minute cooldown
                self.logger.warning("🚨 Circuit breaker activated - too many optimization failures")
                return False
            elif current_time < self._circuit_breaker_reset_time:
                self.logger.debug("Circuit breaker active, skipping optimization")
                return False
            else:
                # Reset circuit breaker
                self._optimization_failures = 0
                self._circuit_breaker_reset_time = None
                self.logger.info("🔄 Circuit breaker reset, resuming optimizations")

        # Enhanced recursion prevention with timeout and stack depth check
        if hasattr(self, '_optimizing') and self._optimizing:
            # Check if we've been optimizing for too long (timeout after 10 seconds)
            if hasattr(self, '_optimization_start_time'):
                if current_time - self._optimization_start_time > 10:
                    self.logger.warning("Optimization timeout detected, clearing recursion guard")
                    self._optimizing = False
                    self._optimization_start_time = None
                else:
                    self.logger.warning("Already optimizing, skipping to prevent recursion")
                    return False
            else:
                self.logger.warning("Already optimizing, skipping to prevent recursion")
                return False
        
        # Additional recursion prevention - check call stack depth
        import sys
        if len(sys._getframe().f_back.f_locals) > 50:  # Arbitrary depth limit
            self.logger.warning("Call stack too deep, skipping optimization to prevent recursion")
            return False
        
        # Set optimization flag and timestamp immediately to prevent recursion
        self._optimizing = True
        self._optimization_start_time = current_time
        optimization_level = optimization_level or self.config.cpu_optimization_level
        self.current_workload_type = workload_type

        try:
            # Reduced logging to prevent verbosity
            if not hasattr(self, '_last_workload') or self._last_workload != workload_type.value:
                self.logger.info(f"🎯 Optimizing for {workload_type.value} workload ({optimization_level.value})")
                self._last_workload = workload_type.value

            # CPU optimization
            self._optimize_cpu_for_workload(workload_type, optimization_level)

            # GPU optimization
            self._optimize_gpu_for_workload(workload_type, optimization_level)

            # Memory optimization
            self._optimize_memory_for_workload(workload_type, optimization_level)

            # Store optimization context
            self.optimization_contexts[workload_type.value] = {
                'optimization_level': optimization_level.value,
                'timestamp': time.time(),
                'cpu_settings': self._get_cpu_settings(),
                'gpu_settings': self._get_gpu_settings(),
                'memory_settings': self._get_memory_settings()
            }

            return True

        except Exception as e:
            self.logger.error(f"Failed to optimize for workload {workload_type.value}: {e}")
            # Track failures for circuit breaker
            self._optimization_failures += 1
            return False
        finally:
            # Clear recursion guard and timestamp
            self._optimizing = False
            self._optimization_start_time = None

    def _optimize_cpu_for_workload(self, workload_type: WorkloadType, level: OptimizationLevel):
        """Optimize CPU for specific workload."""
        if workload_type == WorkloadType.BACKTESTING:
            # Optimize for backtesting - use performance cores
            self.cpu_optimizer.performance_cores = 4
            self.cpu_optimizer.efficiency_cores = 0
        elif workload_type == WorkloadType.ML_TRAINING:
            # Optimize for ML training - balance performance and efficiency
            self.cpu_optimizer.performance_cores = 3
            self.cpu_optimizer.efficiency_cores = 1
        elif workload_type == WorkloadType.DATA_PROCESSING:
            # Optimize for data processing - use all cores
            self.cpu_optimizer.performance_cores = 2
            self.cpu_optimizer.efficiency_cores = 2

    def _optimize_gpu_for_workload(self, workload_type: WorkloadType, level: OptimizationLevel):
        """Optimize GPU for specific workload."""
        if workload_type in [WorkloadType.ML_TRAINING, WorkloadType.MONTE_CARLO]:
            # Enable GPU acceleration for compute-intensive tasks
            if self.gpu_manager.mps_available:
                self.logger.info("🚀 GPU acceleration enabled for compute-intensive workload")
        else:
            # Disable GPU for lighter workloads
            pass

    def _optimize_memory_for_workload(self, workload_type: WorkloadType, level: OptimizationLevel):
        """Optimize memory for specific workload."""
        if workload_type == WorkloadType.ML_TRAINING:
            # Aggressive memory optimization for ML training
            self.memory_optimizer.thresholds['high'] = 0.8
            self.memory_optimizer.thresholds['critical'] = 0.9
        elif workload_type == WorkloadType.BACKTESTING:
            # Moderate memory optimization for backtesting
            self.memory_optimizer.thresholds['high'] = 0.85
            self.memory_optimizer.thresholds['critical'] = 0.95

    def get_optimal_cpu_count(self) -> int:
        """Get optimal CPU count for the current workload."""
        if hasattr(self, 'cpu_optimizer') and self.cpu_optimizer:
            return self.cpu_optimizer.cpu_count
        else:
            # Fallback to basic CPU count
            import multiprocessing
            return max(1, multiprocessing.cpu_count() // 2)

    def _get_cpu_settings(self) -> Dict[str, Any]:
        """Get current CPU settings."""
        return {
            'performance_cores': self.cpu_optimizer.performance_cores,
            'efficiency_cores': self.cpu_optimizer.efficiency_cores,
            'optimal_workers': self.cpu_optimizer.cpu_count
        }

    def _get_gpu_settings(self) -> Dict[str, Any]:
        """Get current GPU settings."""
        return {
            'mps_available': self.gpu_manager.mps_available,
            'gpu_info': self.gpu_manager.get_gpu_info()
        }

    def _get_memory_settings(self) -> Dict[str, Any]:
        """Get current memory settings."""
        return {
            'memory_limit_gb': self.memory_optimizer.memory_limit_gb,
            'thresholds': self.memory_optimizer.thresholds
        }

    def _handle_performance_alert(self, metric_name: str, value: float, threshold: float):
        """Handle performance alerts."""
        # Prevent recursive calls during optimization
        if hasattr(self, '_optimizing') and self._optimizing:
            self.logger.debug(f"🚨 Performance alert during optimization: {metric_name} = {value:.1f} (threshold: {threshold:.1f}) - deferring response")
            return
            
        self.logger.warning(f"🚨 Performance alert: {metric_name} = {value:.1f} (threshold: {threshold:.1f})")

        # Implement adaptive responses with recursion protection
        try:
            if metric_name == 'cpu_usage' and value > 90:
                self._reduce_cpu_intensity()
            elif metric_name == 'memory_usage' and value > 95:
                self._trigger_aggressive_memory_cleanup()
            elif metric_name == 'temperature' and value > 85:
                self._reduce_thermal_load()
        except Exception as e:
            self.logger.error(f"Error handling performance alert: {e}")

    def _reduce_cpu_intensity(self):
        """Reduce CPU intensity in response to high usage."""
        self.logger.info("🔧 Reducing CPU intensity")
        # Reduce thread pool sizes and processing intensity
        if hasattr(self, 'cpu_optimizer'):
            self.cpu_optimizer.set_conservative_mode()

        # Add a small delay to prevent busy waiting
        time.sleep(0.1)

        # Set reduced intensity flag
        self._cpu_intensity_reduced = True

    def _trigger_aggressive_memory_cleanup(self):
        """Trigger aggressive memory cleanup."""
        self.logger.info("🧹 Triggering aggressive memory cleanup")
        self.memory_optimizer._aggressive_memory_cleanup()

    def _reduce_thermal_load(self):
        """Reduce thermal load."""
        self.logger.info("🌡️ Reducing thermal load")
        # Implementation would reduce CPU/GPU frequencies, etc.

    def optimize_for_inference(self):
        """Optimize hardware for inference workload."""
        try:
            # Direct optimization without recursive call to optimize_for_workload
            if not self.is_initialized:
                self.logger.warning("Hardware manager not initialized")
                return False
            
            # Set optimization flag to prevent recursion
            if hasattr(self, '_optimizing') and self._optimizing:
                self.logger.warning("Already optimizing, skipping inference optimization")
                return False
            
            self._optimizing = True
            self._optimization_start_time = time.time()
            
            # Direct optimization for inference
            self._optimize_cpu_for_workload(WorkloadType.ML_TRAINING, OptimizationLevel.AGGRESSIVE)
            self._optimize_gpu_for_workload(WorkloadType.ML_TRAINING, OptimizationLevel.AGGRESSIVE)
            self._optimize_memory_for_workload(WorkloadType.ML_TRAINING, OptimizationLevel.AGGRESSIVE)
            
            self.logger.info("✅ Hardware optimized for inference")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Inference optimization failed: {e}")
            return False
        finally:
            self._optimizing = False
            self._optimization_start_time = None

    @contextmanager
    def optimization_context(self, workload_type: WorkloadType,
                           optimization_level: OptimizationLevel = None):
        """Context manager for workload-specific optimization."""
        optimization_level = optimization_level or self.config.cpu_optimization_level

        # Apply optimization directly without recursive call
        if not self.is_initialized:
            self.logger.warning("Hardware manager not initialized")
            yield self
            return
        
        # Set optimization flag to prevent recursion
        if hasattr(self, '_optimizing') and self._optimizing:
            self.logger.warning("Already optimizing, skipping context optimization")
            yield self
            return
        
        self._optimizing = True
        self._optimization_start_time = time.time()
        
        try:
            # Direct optimization without recursive call
            self._optimize_cpu_for_workload(workload_type, optimization_level)
            self._optimize_gpu_for_workload(workload_type, optimization_level)
            self._optimize_memory_for_workload(workload_type, optimization_level)
            
            yield self
        finally:
            # Clear optimization flag
            self._optimizing = False
            self._optimization_start_time = None

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.is_initialized:
            return {"error": "System not initialized"}

        return {
            "initialized": self.is_initialized,
            "current_workload": self.current_workload_type.value if self.current_workload_type else None,
            "performance_report": self.performance_monitor.get_performance_report(),
            "scheduling_report": self.task_scheduler.get_scheduling_report(),
            "cpu_info": self.cpu_optimizer.get_cpu_info(),
            "gpu_info": self.gpu_manager.get_gpu_info(),
            "memory_stats": self.memory_optimizer.get_memory_stats(),
            "optimization_contexts": self.optimization_contexts,
            "config": {
                "cpu_optimization_level": self.config.cpu_optimization_level.value,
                "gpu_optimization_level": self.config.gpu_optimization_level.value,
                "memory_optimization_level": self.config.memory_optimization_level.value,
                "adaptive_optimization_enabled": self.config.enable_adaptive_optimization
            }
        }

    def save_configuration(self, file_path: str):
        """Save current configuration to file."""
        try:
            config_data = {
                "config": {
                    "cpu_optimization_level": self.config.cpu_optimization_level.value,
                    "gpu_optimization_level": self.config.gpu_optimization_level.value,
                    "memory_optimization_level": self.config.memory_optimization_level.value,
                    "memory_limit_gb": self.config.memory_limit_gb,
                    "enable_adaptive_optimization": self.config.enable_adaptive_optimization,
                    "monitoring_interval": self.config.monitoring_interval,
                    "alert_thresholds": self.config.alert_thresholds
                },
                "optimization_contexts": self.optimization_contexts,
                "timestamp": time.time()
            }

            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)

            self.logger.info(f"💾 Configuration saved to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")

    def load_configuration(self, file_path: str) -> bool:
        """Load configuration from file."""
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)

            # Update configuration
            config_dict = config_data.get("config", {})
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    if key.endswith("_level"):
                        setattr(self.config, key, OptimizationLevel(value))
                    else:
                        setattr(self.config, key, value)

            # Restore optimization contexts
            self.optimization_contexts = config_data.get("optimization_contexts", {})

            self.logger.info(f"📂 Configuration loaded from {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False

# Global instance
_unified_hardware_manager: Optional[UnifiedHardwareManager] = None

def get_unified_hardware_manager(config: Optional[HardwareConfig] = None, conservative_mode: bool = False) -> UnifiedHardwareManager:
    """Get or create the global unified hardware manager instance."""
    global _unified_hardware_manager

    if _unified_hardware_manager is None:
        # Use conservative configuration if requested or if no config provided
        if conservative_mode or config is None:
            config = HardwareConfig(
                cpu_optimization_level=OptimizationLevel.MINIMAL,
                enable_thermal_monitoring=False,
                enable_adaptive_optimization=False,
                monitoring_interval=30.0,
                alert_thresholds={
                    'cpu_usage': 70.0,
                    'memory_usage': 80.0,
                    'gpu_usage': 60.0,
                    'temperature': 70.0
                }
            )

        _unified_hardware_manager = UnifiedHardwareManager(config)
        _unified_hardware_manager.initialize()

    return _unified_hardware_manager

def optimize_for_workload(workload_type: WorkloadType,
                         optimization_level: OptimizationLevel = None) -> bool:
    """Convenience function to optimize for a specific workload."""
    manager = get_unified_hardware_manager()
    return manager.optimize_for_workload(workload_type, optimization_level)

def get_system_status() -> Dict[str, Any]:
    """Get current system status."""
    manager = get_unified_hardware_manager()
    return manager.get_system_status()

def shutdown_hardware_manager():
    """Shutdown the global hardware manager."""
    global _unified_hardware_manager
    if _unified_hardware_manager:
        _unified_hardware_manager.shutdown()
        _unified_hardware_manager = None
