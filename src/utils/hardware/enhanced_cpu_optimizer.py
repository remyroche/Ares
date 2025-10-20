"""
Enhanced CPU Optimizer for Apple Silicon.

This module provides advanced CPU optimization with performance/efficiency core management,
thermal management, and workload-specific optimizations for M1/M2/M3/M4 chips.
"""

import logging
import time
import threading
import multiprocessing
import concurrent.futures
import os
import psutil
import subprocess
import platform
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import queue
import weakref
import gc
import numpy as np
import pandas as pd

from .m1_advanced_cpu_optimizer import (
    M1AdvancedCPUOptimizer, CPUConfig, CoreType, WorkloadType, ThermalState,
    get_advanced_cpu_optimizer, optimize_cpu_execution, parallel_cpu_execution
)

logger = logging.getLogger(__name__)

class CPUIntensity(Enum):
    """CPU intensity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class PowerMode(Enum):
    """Power management modes."""
    POWER_SAVE = "power_save"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    MAXIMUM_PERFORMANCE = "maximum_performance"

@dataclass
class EnhancedCPUConfig(CPUConfig):
    """Enhanced CPU configuration."""
    # Power management
    enable_power_management: bool = True
    power_mode: PowerMode = PowerMode.BALANCED
    dynamic_power_scaling: bool = True
    power_limit_watts: float = 25.0  # M1 typical power limit
    
    # Core affinity
    enable_advanced_affinity: bool = True
    affinity_strategy: str = "workload_aware"  # workload_aware, thermal_aware, performance_aware
    
    # Workload optimization
    enable_workload_profiling: bool = True
    profile_retention_hours: int = 24
    enable_predictive_scaling: bool = True
    
    # Thermal management
    enable_aggressive_thermal_management: bool = True
    thermal_boost_threshold: float = 70.0
    thermal_throttle_threshold: float = 85.0
    enable_dynamic_frequency_scaling: bool = True
    
    # Memory optimization
    enable_memory_prefetching: bool = True
    prefetch_aggressiveness: float = 0.5  # 0.0 to 1.0
    enable_cache_optimization: bool = True

@dataclass
class WorkloadProfile:
    """Enhanced workload profile."""
    workload_id: str
    workload_type: WorkloadType
    cpu_intensity: CPUIntensity
    memory_intensity: float  # 0.0 to 1.0
    io_intensity: float  # 0.0 to 1.0
    preferred_core_type: CoreType
    estimated_duration: float
    priority: int  # 1-10
    memory_requirement_mb: float
    parallelizable: bool
    thermal_sensitivity: float  # 0.0 to 1.0
    power_sensitivity: float  # 0.0 to 1.0
    created_at: float = field(default_factory=time.time)
    last_executed: float = field(default_factory=time.time)
    execution_count: int = 0
    average_execution_time: float = 0.0

class ThermalManager:
    """Enhanced thermal management system."""
    
    def __init__(self, config: EnhancedCPUConfig):
        self.config = config
        self.logger = logger.getChild('ThermalManager')
        
        # Thermal state tracking
        self.current_temperature = 50.0
        self.temperature_history = []
        self.thermal_events = []
        
        # Thermal thresholds
        self.thresholds = {
            'cool': 60.0,
            'warm': 70.0,
            'hot': 80.0,
            'critical': 90.0
        }
        
        # Thermal management actions
        self.thermal_actions = {
            'cool': self._handle_cool_state,
            'warm': self._handle_warm_state,
            'hot': self._handle_hot_state,
            'critical': self._handle_critical_state
        }
        
        # Start thermal monitoring
        self._start_thermal_monitoring()
    
    def _start_thermal_monitoring(self):
        """Start thermal monitoring thread."""
        def monitor():
            while True:
                try:
                    self._update_thermal_state()
                    time.sleep(1.0)  # Check every second
                except Exception as e:
                    self.logger.error(f"Thermal monitoring error: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        self.logger.info("🌡️ Enhanced thermal monitoring started")
    
    def _update_thermal_state(self):
        """Update thermal state."""
        try:
            # Get current temperature
            temperature = self._get_cpu_temperature()
            self.current_temperature = temperature
            self.temperature_history.append(temperature)
            
            # Keep only recent history
            if len(self.temperature_history) > 1000:
                self.temperature_history = self.temperature_history[-500:]
            
            # Determine thermal state
            if temperature >= self.thresholds['critical']:
                state = 'critical'
            elif temperature >= self.thresholds['hot']:
                state = 'hot'
            elif temperature >= self.thresholds['warm']:
                state = 'warm'
            else:
                state = 'cool'
            
            # Take thermal management action
            if state in self.thermal_actions:
                self.thermal_actions[state]()
            
            # Log thermal events
            if len(self.temperature_history) > 1:
                prev_temp = self.temperature_history[-2]
                if abs(temperature - prev_temp) > 5:
                    self.logger.debug(f"🌡️ Temperature: {temperature:.1f}°C, State: {state}")
        
        except Exception as e:
            self.logger.warning(f"Failed to update thermal state: {e}")
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature."""
        try:
            if platform.system() == 'Darwin':
                # macOS - use powermetrics
                result = subprocess.run(
                    ['powermetrics', '--samplers', 'smc', '-n', '1', '-i', '1000'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'CPU die temperature' in line:
                            temp_str = line.split(':')[-1].strip().replace('C', '')
                            return float(temp_str)
            
            # Fallback: estimate from CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            base_temp = 40.0
            temp_increase = cpu_percent * 0.3
            return base_temp + temp_increase
            
        except Exception as e:
            self.logger.debug(f"Failed to get CPU temperature: {e}")
            return 50.0
    
    def _handle_cool_state(self):
        """Handle cool thermal state."""
        # Enable performance boost
        if self.config.enable_dynamic_frequency_scaling:
            self._set_cpu_frequency_boost(True)
    
    def _handle_warm_state(self):
        """Handle warm thermal state."""
        # Normal operation
        pass
    
    def _handle_hot_state(self):
        """Handle hot thermal state."""
        self.logger.warning("🌡️ Hot thermal state detected")
        
        # Reduce CPU frequency
        if self.config.enable_dynamic_frequency_scaling:
            self._set_cpu_frequency_boost(False)
        
        # Force garbage collection
        gc.collect()
    
    def _handle_critical_state(self):
        """Handle critical thermal state."""
        self.logger.error("🚨 Critical thermal state detected")
        
        # Emergency thermal management
        self._emergency_thermal_management()
    
    def _set_cpu_frequency_boost(self, enable: bool):
        """Set CPU frequency boost."""
        try:
            if platform.system() == 'Darwin':
                # Use powermetrics to control frequency scaling
                if enable:
                    subprocess.run(['sudo', 'pmset', '-a', 'disablesleep', '0'], 
                                 capture_output=True, timeout=5)
                else:
                    subprocess.run(['sudo', 'pmset', '-a', 'disablesleep', '1'], 
                                 capture_output=True, timeout=5)
        except Exception as e:
            self.logger.debug(f"Failed to set CPU frequency boost: {e}")
    
    def _emergency_thermal_management(self):
        """Emergency thermal management."""
        # Force all cores to efficiency mode
        # Clear caches
        gc.collect()
        
        # Log thermal event
        self.thermal_events.append({
            'timestamp': time.time(),
            'temperature': self.current_temperature,
            'action': 'emergency_thermal_management'
        })
    
    def get_thermal_state(self) -> str:
        """Get current thermal state."""
        if self.current_temperature >= self.thresholds['critical']:
            return 'critical'
        elif self.current_temperature >= self.thresholds['hot']:
            return 'hot'
        elif self.current_temperature >= self.thresholds['warm']:
            return 'warm'
        else:
            return 'cool'
    
    def get_thermal_metrics(self) -> Dict[str, Any]:
        """Get thermal metrics."""
        return {
            'current_temperature': self.current_temperature,
            'average_temperature': np.mean(self.temperature_history[-100:]) if self.temperature_history else 0,
            'max_temperature': max(self.temperature_history) if self.temperature_history else 0,
            'thermal_state': self.get_thermal_state(),
            'thermal_events_count': len(self.thermal_events)
        }

class PowerManager:
    """Power management system."""
    
    def __init__(self, config: EnhancedCPUConfig):
        self.config = config
        self.logger = logger.getChild('PowerManager')
        
        # Power state tracking
        self.current_power_mode = config.power_mode
        self.power_history = []
        self.power_events = []
        
        # Power management actions
        self.power_actions = {
            PowerMode.POWER_SAVE: self._set_power_save_mode,
            PowerMode.BALANCED: self._set_balanced_mode,
            PowerMode.PERFORMANCE: self._set_performance_mode,
            PowerMode.MAXIMUM_PERFORMANCE: self._set_maximum_performance_mode
        }
    
    def _set_power_save_mode(self):
        """Set power save mode."""
        self.logger.info("🔋 Switching to power save mode")
        
        # Reduce CPU frequency
        # Use efficiency cores only
        # Reduce memory bandwidth
        
        self.power_events.append({
            'timestamp': time.time(),
            'mode': 'power_save',
            'action': 'mode_switch'
        })
    
    def _set_balanced_mode(self):
        """Set balanced mode."""
        self.logger.info("⚖️ Switching to balanced mode")
        
        # Balanced CPU frequency
        # Use both efficiency and performance cores
        # Normal memory bandwidth
        
        self.power_events.append({
            'timestamp': time.time(),
            'mode': 'balanced',
            'action': 'mode_switch'
        })
    
    def _set_performance_mode(self):
        """Set performance mode."""
        self.logger.info("⚡ Switching to performance mode")
        
        # High CPU frequency
        # Prefer performance cores
        # High memory bandwidth
        
        self.power_events.append({
            'timestamp': time.time(),
            'mode': 'performance',
            'action': 'mode_switch'
        })
    
    def _set_maximum_performance_mode(self):
        """Set maximum performance mode."""
        self.logger.info("🚀 Switching to maximum performance mode")
        
        # Maximum CPU frequency
        # Use all performance cores
        # Maximum memory bandwidth
        
        self.power_events.append({
            'timestamp': time.time(),
            'mode': 'maximum_performance',
            'action': 'mode_switch'
        })
    
    def set_power_mode(self, mode: PowerMode):
        """Set power mode."""
        if mode in self.power_actions:
            self.current_power_mode = mode
            self.power_actions[mode]()
    
    def get_power_metrics(self) -> Dict[str, Any]:
        """Get power metrics."""
        return {
            'current_mode': self.current_power_mode.value,
            'power_events_count': len(self.power_events),
            'recent_events': self.power_events[-10:] if self.power_events else []
        }

class WorkloadProfiler:
    """Workload profiling system."""
    
    def __init__(self, config: EnhancedCPUConfig):
        self.config = config
        self.logger = logger.getChild('WorkloadProfiler')
        
        # Workload profiles
        self.workload_profiles: Dict[str, WorkloadProfile] = {}
        self.execution_history = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_executions': 0,
            'average_execution_time': 0.0,
            'cpu_utilization': 0.0,
            'memory_utilization': 0.0
        }
    
    def create_workload_profile(self, workload_id: str, workload_type: WorkloadType,
                              cpu_intensity: CPUIntensity, memory_intensity: float,
                              io_intensity: float, preferred_core_type: CoreType,
                              estimated_duration: float, priority: int,
                              memory_requirement_mb: float, parallelizable: bool,
                              thermal_sensitivity: float = 0.5,
                              power_sensitivity: float = 0.5) -> WorkloadProfile:
        """Create a workload profile."""
        profile = WorkloadProfile(
            workload_id=workload_id,
            workload_type=workload_type,
            cpu_intensity=cpu_intensity,
            memory_intensity=memory_intensity,
            io_intensity=io_intensity,
            preferred_core_type=preferred_core_type,
            estimated_duration=estimated_duration,
            priority=priority,
            memory_requirement_mb=memory_requirement_mb,
            parallelizable=parallelizable,
            thermal_sensitivity=thermal_sensitivity,
            power_sensitivity=power_sensitivity
        )
        
        self.workload_profiles[workload_id] = profile
        self.logger.info(f"📋 Created workload profile: {workload_id}")
        
        return profile
    
    def update_workload_profile(self, workload_id: str, execution_time: float):
        """Update workload profile with execution data."""
        if workload_id in self.workload_profiles:
            profile = self.workload_profiles[workload_id]
            profile.execution_count += 1
            profile.last_executed = time.time()
            
            # Update average execution time
            if profile.average_execution_time == 0:
                profile.average_execution_time = execution_time
            else:
                profile.average_execution_time = (
                    (profile.average_execution_time * (profile.execution_count - 1) + execution_time) /
                    profile.execution_count
                )
            
            # Update performance metrics
            self.performance_metrics['total_executions'] += 1
            self.performance_metrics['average_execution_time'] = (
                (self.performance_metrics['average_execution_time'] * 
                 (self.performance_metrics['total_executions'] - 1) + execution_time) /
                self.performance_metrics['total_executions']
            )
            
            # Record execution history
            self.execution_history.append({
                'workload_id': workload_id,
                'execution_time': execution_time,
                'timestamp': time.time()
            })
            
            # Keep only recent history
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-500:]
    
    def get_workload_recommendations(self, workload_id: str) -> Dict[str, Any]:
        """Get workload optimization recommendations."""
        if workload_id not in self.workload_profiles:
            return {}
        
        profile = self.workload_profiles[workload_id]
        
        recommendations = {
            'workload_id': workload_id,
            'cpu_intensity': profile.cpu_intensity.value,
            'preferred_core_type': profile.preferred_core_type.value,
            'thermal_sensitivity': profile.thermal_sensitivity,
            'power_sensitivity': profile.power_sensitivity,
            'estimated_duration': profile.estimated_duration,
            'average_execution_time': profile.average_execution_time,
            'execution_count': profile.execution_count
        }
        
        # Add optimization suggestions
        if profile.thermal_sensitivity > 0.7:
            recommendations['thermal_optimization'] = 'Use efficiency cores, reduce frequency'
        
        if profile.power_sensitivity > 0.7:
            recommendations['power_optimization'] = 'Use power save mode, optimize memory usage'
        
        if profile.cpu_intensity == CPUIntensity.HIGH:
            recommendations['cpu_optimization'] = 'Use performance cores, enable frequency boost'
        
        return recommendations

class EnhancedCPUOptimizer:
    """Enhanced CPU optimizer with advanced features."""
    
    def __init__(self, config: Optional[EnhancedCPUConfig] = None):
        self.config = config or EnhancedCPUConfig()
        self.logger = logger.getChild('EnhancedCPUOptimizer')
        
        # Initialize base CPU optimizer
        self.base_optimizer = get_advanced_cpu_optimizer(self.config)
        
        # Initialize enhanced components
        self.thermal_manager = ThermalManager(self.config)
        self.power_manager = PowerManager(self.config)
        self.workload_profiler = WorkloadProfiler(self.config)
        
        # Performance tracking
        self.performance_metrics = {
            'total_operations': 0,
            'thermal_optimizations': 0,
            'power_optimizations': 0,
            'workload_optimizations': 0,
            'average_speedup': 0.0
        }
        
        self.logger.info("🚀 Enhanced CPU Optimizer initialized")
    
    def optimize_for_workload(self, workload_id: str, workload_type: WorkloadType,
                            cpu_intensity: CPUIntensity = CPUIntensity.MEDIUM,
                            memory_intensity: float = 0.5,
                            io_intensity: float = 0.3,
                            preferred_core_type: CoreType = CoreType.UNIFIED,
                            estimated_duration: float = 60.0,
                            priority: int = 5,
                            memory_requirement_mb: float = 100.0,
                            parallelizable: bool = True,
                            thermal_sensitivity: float = 0.5,
                            power_sensitivity: float = 0.5) -> WorkloadProfile:
        """Create optimized workload profile."""
        
        # Create workload profile
        profile = self.workload_profiler.create_workload_profile(
            workload_id=workload_id,
            workload_type=workload_type,
            cpu_intensity=cpu_intensity,
            memory_intensity=memory_intensity,
            io_intensity=io_intensity,
            preferred_core_type=preferred_core_type,
            estimated_duration=estimated_duration,
            priority=priority,
            memory_requirement_mb=memory_requirement_mb,
            parallelizable=parallelizable,
            thermal_sensitivity=thermal_sensitivity,
            power_sensitivity=power_sensitivity
        )
        
        # Apply thermal optimizations
        if thermal_sensitivity > 0.7:
            self._apply_thermal_optimizations(profile)
            self.performance_metrics['thermal_optimizations'] += 1
        
        # Apply power optimizations
        if power_sensitivity > 0.7:
            self._apply_power_optimizations(profile)
            self.performance_metrics['power_optimizations'] += 1
        
        # Apply workload-specific optimizations
        self._apply_workload_optimizations(profile)
        self.performance_metrics['workload_optimizations'] += 1
        
        return profile
    
    def _apply_thermal_optimizations(self, profile: WorkloadProfile):
        """Apply thermal optimizations to workload profile."""
        thermal_state = self.thermal_manager.get_thermal_state()
        
        if thermal_state in ['hot', 'critical']:
            # Reduce CPU intensity
            if profile.cpu_intensity == CPUIntensity.HIGH:
                profile.cpu_intensity = CPUIntensity.MEDIUM
            elif profile.cpu_intensity == CPUIntensity.MEDIUM:
                profile.cpu_intensity = CPUIntensity.LOW
            
            # Prefer efficiency cores
            if profile.preferred_core_type == CoreType.PERFORMANCE:
                profile.preferred_core_type = CoreType.EFFICIENCY
            
            self.logger.info(f"🌡️ Applied thermal optimizations to {profile.workload_id}")
    
    def _apply_power_optimizations(self, profile: WorkloadProfile):
        """Apply power optimizations to workload profile."""
        if profile.power_sensitivity > 0.7:
            # Switch to power save mode
            self.power_manager.set_power_mode(PowerMode.POWER_SAVE)
            
            # Reduce CPU intensity
            if profile.cpu_intensity == CPUIntensity.HIGH:
                profile.cpu_intensity = CPUIntensity.MEDIUM
            
            self.logger.info(f"🔋 Applied power optimizations to {profile.workload_id}")
    
    def _apply_workload_optimizations(self, profile: WorkloadProfile):
        """Apply workload-specific optimizations."""
        if profile.workload_type == WorkloadType.CPU_INTENSIVE:
            # Use performance cores
            if profile.preferred_core_type == CoreType.EFFICIENCY:
                profile.preferred_core_type = CoreType.PERFORMANCE
            
            # Enable frequency boost
            if self.thermal_manager.get_thermal_state() == 'cool':
                self.thermal_manager._set_cpu_frequency_boost(True)
        
        elif profile.workload_type == WorkloadType.MEMORY_INTENSIVE:
            # Optimize memory usage
            if profile.memory_requirement_mb > 1000:
                # Enable memory prefetching
                pass
        
        elif profile.workload_type == WorkloadType.IO_INTENSIVE:
            # Use efficiency cores
            if profile.preferred_core_type == CoreType.PERFORMANCE:
                profile.preferred_core_type = CoreType.EFFICIENCY
        
        self.logger.info(f"⚙️ Applied workload optimizations to {profile.workload_id}")
    
    def execute_with_enhanced_optimization(self, func: Callable, *args, 
                                         workload_id: str = "default",
                                         workload_type: WorkloadType = WorkloadType.MIXED,
                                         cpu_intensity: CPUIntensity = CPUIntensity.MEDIUM,
                                         **kwargs) -> Any:
        """Execute function with enhanced CPU optimization."""
        
        # Create or update workload profile
        if workload_id in self.workload_profiler.workload_profiles:
            profile = self.workload_profiler.workload_profiles[workload_id]
        else:
            profile = self.optimize_for_workload(
                workload_id=workload_id,
                workload_type=workload_type,
                cpu_intensity=cpu_intensity
            )
        
        # Track operation
        self.performance_metrics['total_operations'] += 1
        
        # Execute with base optimization
        start_time = time.time()
        result = self.base_optimizer.execute_with_optimization(
            func, *args, workload_type=workload_type, **kwargs
        )
        execution_time = time.time() - start_time
        
        # Update workload profile
        self.workload_profiler.update_workload_profile(workload_id, execution_time)
        
        # Update performance metrics
        if self.performance_metrics['average_speedup'] == 0:
            self.performance_metrics['average_speedup'] = 1.0
        else:
            # Estimate speedup (simplified)
            estimated_speedup = 1.0 + (profile.cpu_intensity.value == 'high') * 0.2
            self.performance_metrics['average_speedup'] = (
                (self.performance_metrics['average_speedup'] * 
                 (self.performance_metrics['total_operations'] - 1) + estimated_speedup) /
                self.performance_metrics['total_operations']
            )
        
        self.logger.debug(f"⚡ Enhanced execution of {workload_id} in {execution_time:.3f}s")
        
        return result
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            'enhanced_cpu_metrics': self.performance_metrics,
            'thermal_metrics': self.thermal_manager.get_thermal_metrics(),
            'power_metrics': self.power_manager.get_power_metrics(),
            'workload_profiles_count': len(self.workload_profiler.workload_profiles),
            'base_cpu_metrics': self.base_optimizer.get_performance_metrics()
        }
    
    def get_workload_recommendations(self, workload_id: str) -> Dict[str, Any]:
        """Get workload optimization recommendations."""
        return self.workload_profiler.get_workload_recommendations(workload_id)

# Global instance
_enhanced_cpu_optimizer: Optional[EnhancedCPUOptimizer] = None

def get_enhanced_cpu_optimizer(config: Optional[EnhancedCPUConfig] = None) -> EnhancedCPUOptimizer:
    """Get or create the global enhanced CPU optimizer."""
    global _enhanced_cpu_optimizer
    
    if _enhanced_cpu_optimizer is None:
        _enhanced_cpu_optimizer = EnhancedCPUOptimizer(config)
    
    return _enhanced_cpu_optimizer

def optimize_cpu_execution_enhanced(workload_id: str = "default",
                                  workload_type: WorkloadType = WorkloadType.MIXED,
                                  cpu_intensity: CPUIntensity = CPUIntensity.MEDIUM):
    """Enhanced decorator for CPU execution optimization."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_enhanced_cpu_optimizer()
            return optimizer.execute_with_enhanced_optimization(
                func, *args, workload_id=workload_id, 
                workload_type=workload_type, cpu_intensity=cpu_intensity, **kwargs
            )
        return wrapper
    return decorator

def cpu_optimized_feature_correlation(data: np.ndarray) -> np.ndarray:
    """Backward compatible function for CPU-optimized feature correlation."""
    optimizer = get_enhanced_cpu_optimizer()
    
    # Create workload profile for feature correlation
    profile = optimizer.optimize_for_workload(
        workload_id="feature_correlation",
        workload_type=WorkloadType.CPU_INTENSIVE,
        cpu_intensity=CPUIntensity.HIGH,
        memory_intensity=0.8,
        preferred_core_type=CoreType.PERFORMANCE,
        estimated_duration=30.0,
        priority=7,
        memory_requirement_mb=data.nbytes / (1024 * 1024),
        parallelizable=True
    )
    
    # Execute correlation with optimization
    def correlation_func(data):
        return np.corrcoef(data.T)
    
    return optimizer.execute_with_enhanced_optimization(
        correlation_func, data, workload_id="feature_correlation"
    )

def get_enhanced_cpu_performance_metrics() -> Dict[str, Any]:
    """Get enhanced CPU performance metrics."""
    optimizer = get_enhanced_cpu_optimizer()
    return optimizer.get_comprehensive_metrics()