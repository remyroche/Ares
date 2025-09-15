"""
Advanced CPU Optimizer for Apple Silicon.

This module extends the basic M1CPUOptimizer with advanced features including
core affinity management, thermal monitoring, power management, and workload-specific optimizations.
"""

import logging
import multiprocessing
import concurrent.futures
import threading
import time
import subprocess
import os
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import platform
import queue
import signal
import sys

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from .m1_cpu_optimizer import M1CPUOptimizer

logger = logging.getLogger(__name__)

class CoreType(Enum):
    """Types of CPU cores."""
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    ALL = "all"

class ThermalState(Enum):
    """Thermal states."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    THROTTLING = "throttling"

class PowerState(Enum):
    """Power states."""
    HIGH_PERFORMANCE = "high_performance"
    BALANCED = "balanced"
    POWER_SAVER = "power_saver"
    ADAPTIVE = "adaptive"

@dataclass
class CoreAffinityConfig:
    """Configuration for core affinity management."""
    enable_affinity: bool = True
    performance_core_mask: int = 0xF  # First 4 cores (performance cores)
    efficiency_core_mask: int = 0xF0  # Next 4 cores (efficiency cores)
    auto_affinity: bool = True
    affinity_timeout: float = 30.0

@dataclass
class ThermalConfig:
    """Configuration for thermal monitoring."""
    enable_monitoring: bool = True
    monitoring_interval: float = 2.0
    warning_threshold: float = 75.0  # Celsius
    critical_threshold: float = 85.0  # Celsius
    throttling_threshold: float = 90.0  # Celsius
    auto_throttling: bool = True

@dataclass
class PowerConfig:
    """Configuration for power management."""
    enable_power_management: bool = True
    power_state: PowerState = PowerState.ADAPTIVE
    max_power_limit: float = 100.0  # Percentage
    auto_power_scaling: bool = True
    power_monitoring_interval: float = 5.0

@dataclass
class WorkloadProfile:
    """Profile for workload-specific optimization."""
    name: str
    cpu_intensity: float  # 0.0 to 1.0
    memory_intensity: float  # 0.0 to 1.0
    thermal_sensitivity: float  # 0.0 to 1.0
    power_sensitivity: float  # 0.0 to 1.0
    preferred_cores: CoreType = CoreType.ALL
    max_threads: Optional[int] = None
    priority: int = 5  # 1-10, higher is more important

class CoreAffinityManager:
    """Manages CPU core affinity for optimal performance."""
    
    def __init__(self, config: CoreAffinityConfig):
        self.config = config
        self.logger = logger.getChild('CoreAffinityManager')
        self.original_affinities: Dict[int, Any] = {}
        self.active_affinities: Dict[int, Any] = {}
        
    def set_core_affinity(self, process_id: int, core_mask: int) -> bool:
        """Set core affinity for a process."""
        try:
            if not self.config.enable_affinity:
                return True
                
            # Store original affinity
            if process_id not in self.original_affinities:
                try:
                    original = os.sched_getaffinity(process_id)
                    self.original_affinities[process_id] = original
                except Exception:
                    self.original_affinities[process_id] = None
                    
            # Set new affinity
            os.sched_setaffinity(process_id, {i for i in range(8) if (core_mask >> i) & 1})
            self.active_affinities[process_id] = core_mask
            
            self.logger.debug(f"Set core affinity for PID {process_id}: {bin(core_mask)}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to set core affinity for PID {process_id}: {e}")
            return False
            
    def restore_affinity(self, process_id: int) -> bool:
        """Restore original core affinity for a process."""
        try:
            if process_id in self.original_affinities and self.original_affinities[process_id]:
                os.sched_setaffinity(process_id, self.original_affinities[process_id])
                del self.original_affinities[process_id]
                
            if process_id in self.active_affinities:
                del self.active_affinities[process_id]
                
            self.logger.debug(f"Restored core affinity for PID {process_id}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to restore core affinity for PID {process_id}: {e}")
            return False
            
    def get_optimal_cores_for_workload(self, workload_profile: WorkloadProfile) -> int:
        """Get optimal core mask for workload profile."""
        if workload_profile.preferred_cores == CoreType.PERFORMANCE:
            return self.config.performance_core_mask
        elif workload_profile.preferred_cores == CoreType.EFFICIENCY:
            return self.config.efficiency_core_mask
        else:
            # Auto-select based on workload characteristics
            if workload_profile.cpu_intensity > 0.7:
                return self.config.performance_core_mask
            elif workload_profile.thermal_sensitivity > 0.7:
                return self.config.efficiency_core_mask
            else:
                return self.config.performance_core_mask | self.config.efficiency_core_mask

class ThermalMonitor:
    """Monitors CPU temperature and thermal states."""
    
    def __init__(self, config: ThermalConfig):
        self.config = config
        self.logger = logger.getChild('ThermalMonitor')
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.current_temperature = 45.0
        self.thermal_state = ThermalState.NORMAL
        self.temperature_history: List[Tuple[float, float]] = []  # (timestamp, temperature)
        self.thermal_callbacks: List[Callable] = []
        
    def start_monitoring(self):
        """Start thermal monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("🌡️ Thermal monitoring started")
        
    def stop_monitoring(self):
        """Stop thermal monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        self.logger.info("🌡️ Thermal monitoring stopped")
        
    def _monitoring_loop(self):
        """Main thermal monitoring loop."""
        while self.monitoring_active:
            try:
                temperature = self._get_cpu_temperature()
                self.current_temperature = temperature
                
                # Update thermal state
                old_state = self.thermal_state
                self.thermal_state = self._determine_thermal_state(temperature)
                
                # Record temperature history
                self.temperature_history.append((time.time(), temperature))
                
                # Keep only recent history (last hour)
                cutoff_time = time.time() - 3600
                self.temperature_history = [
                    (ts, temp) for ts, temp in self.temperature_history 
                    if ts > cutoff_time
                ]
                
                # Trigger callbacks if state changed
                if old_state != self.thermal_state:
                    self._trigger_thermal_callbacks(old_state, self.thermal_state, temperature)
                    
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Thermal monitoring error: {e}")
                time.sleep(5)
                
    def _get_cpu_temperature(self) -> float:
        """Get current CPU temperature."""
        try:
            if platform.system() == 'Darwin':
                # Try to get temperature from powermetrics
                result = subprocess.run(
                    ['sudo', 'powermetrics', '--samplers', 'smc', '-n', '1', '-i', '1000'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    # Parse temperature from output
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'CPU die temperature' in line:
                            try:
                                temp_str = line.split(':')[1].strip().split()[0]
                                return float(temp_str)
                            except (IndexError, ValueError):
                                pass
                                
            # Fallback: estimate based on CPU usage
            if PSUTIL_AVAILABLE:
                cpu_usage = psutil.cpu_percent(interval=1)
                base_temp = 35.0
                temp_increase = cpu_usage * 0.3  # Rough estimate
                return base_temp + temp_increase
            else:
                return 45.0  # Default fallback
            
        except Exception as e:
            self.logger.debug(f"Failed to get CPU temperature: {e}")
            return 45.0  # Default fallback
            
    def _determine_thermal_state(self, temperature: float) -> ThermalState:
        """Determine thermal state based on temperature."""
        if temperature >= self.config.throttling_threshold:
            return ThermalState.THROTTLING
        elif temperature >= self.config.critical_threshold:
            return ThermalState.CRITICAL
        elif temperature >= self.config.warning_threshold:
            return ThermalState.WARNING
        else:
            return ThermalState.NORMAL
            
    def _trigger_thermal_callbacks(self, old_state: ThermalState, new_state: ThermalState, temperature: float):
        """Trigger thermal state change callbacks."""
        self.logger.info(f"🌡️ Thermal state changed: {old_state.value} -> {new_state.value} ({temperature:.1f}°C)")
        
        for callback in self.thermal_callbacks:
            try:
                callback(old_state, new_state, temperature)
            except Exception as e:
                self.logger.error(f"Thermal callback error: {e}")
                
    def add_thermal_callback(self, callback: Callable):
        """Add thermal state change callback."""
        self.thermal_callbacks.append(callback)
        
    def get_thermal_stats(self) -> Dict[str, Any]:
        """Get thermal statistics."""
        if not self.temperature_history:
            return {"error": "No temperature data available"}
            
        temperatures = [temp for _, temp in self.temperature_history]
        
        return {
            "current_temperature": self.current_temperature,
            "thermal_state": self.thermal_state.value,
            "average_temperature": sum(temperatures) / len(temperatures),
            "max_temperature": max(temperatures),
            "min_temperature": min(temperatures),
            "temperature_trend": self._calculate_temperature_trend(),
            "monitoring_duration_minutes": (time.time() - self.temperature_history[0][0]) / 60 if self.temperature_history else 0
        }
        
    def _calculate_temperature_trend(self) -> str:
        """Calculate temperature trend."""
        if len(self.temperature_history) < 2:
            return "unknown"
            
        recent_temps = [temp for _, temp in self.temperature_history[-10:]]
        if len(recent_temps) < 2:
            return "unknown"
            
        # Simple linear trend
        first_half = sum(recent_temps[:len(recent_temps)//2]) / (len(recent_temps)//2)
        second_half = sum(recent_temps[len(recent_temps)//2:]) / (len(recent_temps) - len(recent_temps)//2)
        
        if second_half > first_half + 2:
            return "rising"
        elif second_half < first_half - 2:
            return "falling"
        else:
            return "stable"

class PowerManager:
    """Manages CPU power consumption and scaling."""
    
    def __init__(self, config: PowerConfig):
        self.config = config
        self.logger = logger.getChild('PowerManager')
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.current_power_state = config.power_state
        self.power_history: List[Tuple[float, float]] = []  # (timestamp, power_usage)
        self.power_callbacks: List[Callable] = []
        
    def start_monitoring(self):
        """Start power monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("⚡ Power monitoring started")
        
    def stop_monitoring(self):
        """Stop power monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        self.logger.info("⚡ Power monitoring stopped")
        
    def _monitoring_loop(self):
        """Main power monitoring loop."""
        while self.monitoring_active:
            try:
                power_usage = self._get_power_usage()
                
                # Record power history
                self.power_history.append((time.time(), power_usage))
                
                # Keep only recent history (last hour)
                cutoff_time = time.time() - 3600
                self.power_history = [
                    (ts, power) for ts, power in self.power_history 
                    if ts > cutoff_time
                ]
                
                # Auto power scaling if enabled
                if self.config.auto_power_scaling:
                    self._auto_power_scaling(power_usage)
                    
                time.sleep(self.config.power_monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Power monitoring error: {e}")
                time.sleep(10)
                
    def _get_power_usage(self) -> float:
        """Get current power usage."""
        try:
            if platform.system() == 'Darwin':
                # Try to get power usage from powermetrics
                result = subprocess.run(
                    ['sudo', 'powermetrics', '--samplers', 'cpu_power', '-n', '1', '-i', '1000'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    # Parse power from output
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'CPU Power' in line:
                            try:
                                power_str = line.split(':')[1].strip().split()[0]
                                return float(power_str)
                            except (IndexError, ValueError):
                                pass
                                
            # Fallback: estimate based on CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            base_power = 5.0  # Base power consumption in watts
            power_increase = cpu_usage * 0.1  # Rough estimate
            return base_power + power_increase
            
        except Exception as e:
            self.logger.debug(f"Failed to get power usage: {e}")
            return 10.0  # Default fallback
            
    def _auto_power_scaling(self, current_power: float):
        """Automatically adjust power scaling based on usage."""
        try:
            # Simple power scaling logic
            if current_power > 20.0:  # High power usage
                if self.current_power_state != PowerState.POWER_SAVER:
                    self.set_power_state(PowerState.POWER_SAVER)
            elif current_power < 8.0:  # Low power usage
                if self.current_power_state != PowerState.HIGH_PERFORMANCE:
                    self.set_power_state(PowerState.HIGH_PERFORMANCE)
            else:  # Medium power usage
                if self.current_power_state != PowerState.BALANCED:
                    self.set_power_state(PowerState.BALANCED)
                    
        except Exception as e:
            self.logger.error(f"Auto power scaling error: {e}")
            
    def set_power_state(self, power_state: PowerState):
        """Set power state."""
        old_state = self.current_power_state
        self.current_power_state = power_state
        
        self.logger.info(f"⚡ Power state changed: {old_state.value} -> {power_state.value}")
        
        # Trigger callbacks
        for callback in self.power_callbacks:
            try:
                callback(old_state, power_state)
            except Exception as e:
                self.logger.error(f"Power callback error: {e}")
                
    def add_power_callback(self, callback: Callable):
        """Add power state change callback."""
        self.power_callbacks.append(callback)
        
    def get_power_stats(self) -> Dict[str, Any]:
        """Get power statistics."""
        if not self.power_history:
            return {"error": "No power data available"}
            
        power_values = [power for _, power in self.power_history]
        
        return {
            "current_power_state": self.current_power_state.value,
            "average_power": sum(power_values) / len(power_values),
            "max_power": max(power_values),
            "min_power": min(power_values),
            "power_trend": self._calculate_power_trend(),
            "monitoring_duration_minutes": (time.time() - self.power_history[0][0]) / 60 if self.power_history else 0
        }
        
    def _calculate_power_trend(self) -> str:
        """Calculate power trend."""
        if len(self.power_history) < 2:
            return "unknown"
            
        recent_power = [power for _, power in self.power_history[-10:]]
        if len(recent_power) < 2:
            return "unknown"
            
        # Simple linear trend
        first_half = sum(recent_power[:len(recent_power)//2]) / (len(recent_power)//2)
        second_half = sum(recent_power[len(recent_power)//2:]) / (len(recent_power) - len(recent_power)//2)
        
        if second_half > first_half + 1:
            return "rising"
        elif second_half < first_half - 1:
            return "falling"
        else:
            return "stable"

class AdvancedM1CPUOptimizer(M1CPUOptimizer):
    """Advanced M1 CPU optimizer with enhanced features."""
    
    def __init__(self, 
                 core_affinity_config: Optional[CoreAffinityConfig] = None,
                 thermal_config: Optional[ThermalConfig] = None,
                 power_config: Optional[PowerConfig] = None):
        super().__init__()
        
        # Initialize advanced components
        self.core_affinity_config = core_affinity_config or CoreAffinityConfig()
        self.thermal_config = thermal_config or ThermalConfig()
        self.power_config = power_config or PowerConfig()
        
        self.core_affinity_manager = CoreAffinityManager(self.core_affinity_config)
        self.thermal_monitor = ThermalMonitor(self.thermal_config)
        self.power_manager = PowerManager(self.power_config)
        
        # Workload profiles
        self.workload_profiles: Dict[str, WorkloadProfile] = {
            'backtesting': WorkloadProfile(
                name='backtesting',
                cpu_intensity=0.8,
                memory_intensity=0.6,
                thermal_sensitivity=0.4,
                power_sensitivity=0.3,
                preferred_cores=CoreType.PERFORMANCE,
                max_threads=4
            ),
            'ml_training': WorkloadProfile(
                name='ml_training',
                cpu_intensity=0.9,
                memory_intensity=0.9,
                thermal_sensitivity=0.8,
                power_sensitivity=0.7,
                preferred_cores=CoreType.ALL,
                max_threads=8
            ),
            'data_processing': WorkloadProfile(
                name='data_processing',
                cpu_intensity=0.6,
                memory_intensity=0.8,
                thermal_sensitivity=0.3,
                power_sensitivity=0.4,
                preferred_cores=CoreType.EFFICIENCY,
                max_threads=6
            ),
            'monte_carlo': WorkloadProfile(
                name='monte_carlo',
                cpu_intensity=0.7,
                memory_intensity=0.5,
                thermal_sensitivity=0.5,
                power_sensitivity=0.4,
                preferred_cores=CoreType.PERFORMANCE,
                max_threads=4
            )
        }
        
        # Set up callbacks
        self.thermal_monitor.add_thermal_callback(self._handle_thermal_state_change)
        self.power_manager.add_power_callback(self._handle_power_state_change)
        
        self.logger = logger.getChild('AdvancedM1CPUOptimizer')
        self.logger.info("🚀 Advanced M1 CPU Optimizer initialized")
        
    def start_advanced_monitoring(self):
        """Start all advanced monitoring."""
        if self.thermal_config.enable_monitoring:
            self.thermal_monitor.start_monitoring()
        if self.power_config.enable_power_management:
            self.power_manager.start_monitoring()
        self.logger.info("🔍 Advanced monitoring started")
        
    def stop_advanced_monitoring(self):
        """Stop all advanced monitoring."""
        self.thermal_monitor.stop_monitoring()
        self.power_manager.stop_monitoring()
        self.logger.info("🔍 Advanced monitoring stopped")
        
    def optimize_for_workload_profile(self, profile_name: str) -> bool:
        """Optimize CPU for specific workload profile."""
        if profile_name not in self.workload_profiles:
            self.logger.warning(f"Unknown workload profile: {profile_name}")
            return False
            
        profile = self.workload_profiles[profile_name]
        self.logger.info(f"🎯 Optimizing for workload profile: {profile.name}")
        
        try:
            # Set core affinity
            if self.core_affinity_config.enable_affinity:
                core_mask = self.core_affinity_manager.get_optimal_cores_for_workload(profile)
                self.core_affinity_manager.set_core_affinity(os.getpid(), core_mask)
                
            # Adjust thread pool sizes
            if profile.max_threads:
                self.cpu_count = min(profile.max_threads, self.cpu_count)
                
            # Adjust power state based on workload
            if profile.power_sensitivity > 0.7:
                self.power_manager.set_power_state(PowerState.POWER_SAVER)
            elif profile.cpu_intensity > 0.8:
                self.power_manager.set_power_state(PowerState.HIGH_PERFORMANCE)
            else:
                self.power_manager.set_power_state(PowerState.BALANCED)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to optimize for workload profile {profile_name}: {e}")
            return False
            
    def create_optimized_thread_pool_with_affinity(self, 
                                                  max_workers: Optional[int] = None,
                                                  workload_profile: Optional[str] = None) -> concurrent.futures.ThreadPoolExecutor:
        """Create thread pool with core affinity optimization."""
        if max_workers is None:
            max_workers = self.cpu_count
            
        # Apply workload-specific optimization
        if workload_profile and workload_profile in self.workload_profiles:
            profile = self.workload_profiles[workload_profile]
            max_workers = min(max_workers, profile.max_threads or max_workers)
            
        # Create thread pool with custom thread factory
        def thread_factory():
            thread = threading.Thread()
            # Set thread name for identification
            thread.name = f"M1-Optimized-{threading.current_thread().ident}"
            return thread
            
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix='M1-Advanced'
        )
        
        self.logger.info(f"🧵 Created optimized thread pool with {max_workers} workers")
        return executor
        
    def _handle_thermal_state_change(self, old_state: ThermalState, new_state: ThermalState, temperature: float):
        """Handle thermal state changes."""
        if new_state == ThermalState.CRITICAL:
            self.logger.warning("🌡️ Critical temperature reached, reducing CPU intensity")
            self._reduce_cpu_intensity()
        elif new_state == ThermalState.THROTTLING:
            self.logger.warning("🌡️ CPU throttling detected, switching to efficiency cores")
            self._switch_to_efficiency_cores()
            
    def _handle_power_state_change(self, old_state: PowerState, new_state: PowerState):
        """Handle power state changes."""
        self.logger.info(f"⚡ Power state changed to {new_state.value}")
        
        if new_state == PowerState.POWER_SAVER:
            self._reduce_cpu_intensity()
        elif new_state == PowerState.HIGH_PERFORMANCE:
            self._increase_cpu_intensity()
            
    def _reduce_cpu_intensity(self):
        """Reduce CPU intensity to manage thermal/power constraints."""
        # Reduce thread pool sizes
        self.cpu_count = max(1, self.cpu_count // 2)
        
        # Switch to efficiency cores if available
        if self.efficiency_cores > 0:
            self.performance_cores = 0
            self.efficiency_cores = min(4, self.efficiency_cores)
            
        self.logger.info("🔧 CPU intensity reduced")
        
    def _increase_cpu_intensity(self):
        """Increase CPU intensity for high performance."""
        # Restore original thread pool sizes
        self.cpu_count = self._get_optimal_cpu_count()
        
        # Use performance cores
        self.performance_cores = 4
        self.efficiency_cores = max(0, multiprocessing.cpu_count() - 4)
        
        self.logger.info("🔧 CPU intensity increased")
        
    def _switch_to_efficiency_cores(self):
        """Switch to efficiency cores for thermal management."""
        if self.efficiency_cores > 0:
            self.performance_cores = 0
            self.cpu_count = self.efficiency_cores
            self.logger.info("🔄 Switched to efficiency cores")
            
    def get_advanced_cpu_info(self) -> Dict[str, Any]:
        """Get advanced CPU information."""
        base_info = self.get_cpu_info()
        
        return {
            **base_info,
            "thermal_stats": self.thermal_monitor.get_thermal_stats(),
            "power_stats": self.power_manager.get_power_stats(),
            "core_affinity_config": {
                "enable_affinity": self.core_affinity_config.enable_affinity,
                "performance_core_mask": bin(self.core_affinity_config.performance_core_mask),
                "efficiency_core_mask": bin(self.core_affinity_config.efficiency_core_mask)
            },
            "workload_profiles": {
                name: {
                    "cpu_intensity": profile.cpu_intensity,
                    "memory_intensity": profile.memory_intensity,
                    "thermal_sensitivity": profile.thermal_sensitivity,
                    "power_sensitivity": profile.power_sensitivity,
                    "preferred_cores": profile.preferred_cores.value,
                    "max_threads": profile.max_threads
                }
                for name, profile in self.workload_profiles.items()
            }
        }
        
    def add_workload_profile(self, profile: WorkloadProfile):
        """Add a custom workload profile."""
        self.workload_profiles[profile.name] = profile
        self.logger.info(f"📋 Added workload profile: {profile.name}")
        
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current state."""
        recommendations = []
        
        # Thermal recommendations
        thermal_stats = self.thermal_monitor.get_thermal_stats()
        if thermal_stats.get("thermal_state") == "warning":
            recommendations.append("Consider reducing CPU intensity due to high temperature")
        elif thermal_stats.get("thermal_state") == "critical":
            recommendations.append("Switch to efficiency cores immediately due to critical temperature")
            
        # Power recommendations
        power_stats = self.power_manager.get_power_stats()
        if power_stats.get("average_power", 0) > 15:
            recommendations.append("Consider power saver mode for better efficiency")
            
        # Performance recommendations
        if self.cpu_count < 4:
            recommendations.append("Consider increasing thread count for better performance")
            
        return recommendations

# Global instance
_advanced_cpu_optimizer: Optional[AdvancedM1CPUOptimizer] = None

def get_advanced_cpu_optimizer() -> AdvancedM1CPUOptimizer:
    """Get the global advanced CPU optimizer instance."""
    global _advanced_cpu_optimizer
    
    if _advanced_cpu_optimizer is None:
        _advanced_cpu_optimizer = AdvancedM1CPUOptimizer()
        _advanced_cpu_optimizer.start_advanced_monitoring()
        
    return _advanced_cpu_optimizer

def optimize_for_workload_profile(profile_name: str) -> bool:
    """Convenience function to optimize for a workload profile."""
    optimizer = get_advanced_cpu_optimizer()
    return optimizer.optimize_for_workload_profile(profile_name)

def get_advanced_cpu_info() -> Dict[str, Any]:
    """Get advanced CPU information."""
    optimizer = get_advanced_cpu_optimizer()
    return optimizer.get_advanced_cpu_info()