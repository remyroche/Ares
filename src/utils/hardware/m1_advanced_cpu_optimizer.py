"""
M1 Advanced CPU Optimizer for Apple Silicon.

This module provides advanced CPU optimization techniques specifically designed
for M1/M2/M3/M4 performance and efficiency cores, including thread affinity,
workload distribution, and thermal management.
"""

import logging
import multiprocessing
import concurrent.futures
import threading
import time
import os
import asyncio
import subprocess
import platform
import psutil
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import queue
import weakref
import gc

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_performance, LogLevel
)

logger = logging.getLogger(__name__)

class CoreType(Enum):
    """Types of CPU cores."""
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    UNIFIED = "unified"

class WorkloadType(Enum):
    """Types of workloads for CPU optimization."""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    MIXED = "mixed"
    BATCH = "batch"
    STREAMING = "streaming"

class ThermalState(Enum):
    """Thermal states for thermal management."""
    COOL = "cool"
    WARM = "warm"
    HOT = "hot"
    THROTTLED = "throttled"

@dataclass
class CoreInfo:
    """Information about a CPU core."""
    core_id: int
    core_type: CoreType
    max_frequency_ghz: float
    current_frequency_ghz: float
    temperature_c: float
    utilization_percent: float
    is_available: bool = True
    thread_affinity: Optional[int] = None

@dataclass
class CPUConfig:
    """Configuration for CPU optimization."""
    # Core management
    enable_core_affinity: bool = True
    enable_thermal_management: bool = True
    enable_dynamic_scaling: bool = True
    
    # Performance cores
    performance_cores_count: int = 4
    performance_core_priority: int = 1
    
    # Efficiency cores
    efficiency_cores_count: int = 4
    efficiency_core_priority: int = 2
    
    # Thermal management
    thermal_check_interval: float = 1.0
    thermal_threshold_c: float = 80.0
    throttling_threshold_c: float = 90.0
    
    # Workload distribution
    enable_workload_balancing: bool = True
    balance_check_interval: float = 5.0
    load_balance_threshold: float = 0.2
    
    # Thread management
    max_threads_per_core: int = 2
    thread_pool_size: int = 8
    enable_thread_pooling: bool = True
    
    # Power management
    enable_power_management: bool = True
    power_save_mode: bool = False
    dynamic_frequency_scaling: bool = True

@dataclass
class WorkloadProfile:
    """Profile for a specific workload."""
    workload_type: WorkloadType
    cpu_intensity: float  # 0.0 to 1.0
    memory_intensity: float  # 0.0 to 1.0
    io_intensity: float  # 0.0 to 1.0
    preferred_core_type: CoreType
    estimated_duration: float  # seconds
    priority: int  # 1-10, higher is more important
    memory_requirement_mb: float = 0.0
    parallelizable: bool = True

class ThermalManager:
    """Manages CPU thermal state and throttling."""
    
    def __init__(self, config: CPUConfig):
        self.config = config
        self.logger = logger.getChild('ThermalManager')
        self.current_state = ThermalState.COOL
        self.temperature_history = []
        self.throttling_active = False
        
        # Start thermal monitoring
        if self.config.enable_thermal_management:
            self._start_thermal_monitoring()
    
    def _start_thermal_monitoring(self):
        """Start thermal monitoring thread."""
        def monitor():
            while True:
                try:
                    self._check_thermal_state()
                    time.sleep(self.config.thermal_check_interval)
                except Exception as e:
                    self.logger.error(f"Thermal monitoring error: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        self.logger.info("🌡️ Thermal monitoring started")
    
    def _check_thermal_state(self):
        """Check current thermal state."""
        try:
            # Get CPU temperature (simplified - in real implementation would use proper sensors)
            temperature = self._get_cpu_temperature()
            self.temperature_history.append(temperature)
            
            # Keep only recent history
            if len(self.temperature_history) > 100:
                self.temperature_history.pop(0)
            
            # Determine thermal state
            if temperature >= self.config.throttling_threshold_c:
                self.current_state = ThermalState.THROTTLED
                self.throttling_active = True
            elif temperature >= self.config.thermal_threshold_c:
                self.current_state = ThermalState.HOT
                self.throttling_active = False
            elif temperature >= 60:
                self.current_state = ThermalState.WARM
                self.throttling_active = False
            else:
                self.current_state = ThermalState.COOL
                self.throttling_active = False
            
            # Log state changes
            if len(self.temperature_history) > 1:
                prev_temp = self.temperature_history[-2]
                if abs(temperature - prev_temp) > 5:
                    self.logger.debug(f"🌡️ Temperature: {temperature:.1f}°C, State: {self.current_state.value}")
        
        except Exception as e:
            self.logger.warning(f"Failed to check thermal state: {e}")
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature (simplified implementation)."""
        try:
            # Try to get temperature from system
            if platform.system() == 'Darwin':
                # macOS - try to get temperature from powermetrics
                result = subprocess.run(
                    ['powermetrics', '--samplers', 'smc', '-n', '1', '-i', '1000'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Parse temperature from powermetrics output
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'CPU die temperature' in line:
                            temp_str = line.split(':')[-1].strip().replace('C', '')
                            return float(temp_str)
            
            # Fallback: estimate based on CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            base_temp = 40.0  # Base temperature
            temp_increase = cpu_percent * 0.5  # 0.5°C per 1% CPU usage
            return base_temp + temp_increase
            
        except Exception as e:
            self.logger.debug(f"Failed to get CPU temperature: {e}")
            return 50.0  # Default temperature
    
    def get_thermal_state(self) -> ThermalState:
        """Get current thermal state."""
        return self.current_state
    
    def is_throttling_active(self) -> bool:
        """Check if CPU throttling is active."""
        return self.throttling_active
    
    def get_temperature_history(self) -> List[float]:
        """Get temperature history."""
        return self.temperature_history.copy()

class CoreManager:
    """Manages CPU cores and thread affinity."""
    
    def __init__(self, config: CPUConfig):
        self.config = config
        self.logger = logger.getChild('CoreManager')
        
        # Detect M1 cores
        self.cores = self._detect_cores()
        self.core_utilization = {core.core_id: 0.0 for core in self.cores}
        self.thread_assignments = {}
        
        # Initialize core priorities
        self._initialize_core_priorities()
        
        self.logger.info(f"🔧 Detected {len(self.cores)} cores: "
                        f"{len([c for c in self.cores if c.core_type == CoreType.PERFORMANCE])} performance, "
                        f"{len([c for c in self.cores if c.core_type == CoreType.EFFICIENCY])} efficiency")
    
    def _detect_cores(self) -> List[CoreInfo]:
        """Detect available CPU cores."""
        cores = []
        cpu_count = multiprocessing.cpu_count()
        
        # M1 typically has 4 performance cores and 4 efficiency cores
        # This is a simplified detection - real implementation would use proper system calls
        performance_cores = min(4, cpu_count // 2)
        efficiency_cores = cpu_count - performance_cores
        
        # Create performance cores
        for i in range(performance_cores):
            cores.append(CoreInfo(
                core_id=i,
                core_type=CoreType.PERFORMANCE,
                max_frequency_ghz=3.2,  # M1 performance core frequency
                current_frequency_ghz=3.2,
                temperature_c=45.0,
                utilization_percent=0.0
            ))
        
        # Create efficiency cores
        for i in range(performance_cores, performance_cores + efficiency_cores):
            cores.append(CoreInfo(
                core_id=i,
                core_type=CoreType.EFFICIENCY,
                max_frequency_ghz=2.0,  # M1 efficiency core frequency
                current_frequency_ghz=2.0,
                temperature_c=40.0,
                utilization_percent=0.0
            ))
        
        return cores
    
    def _initialize_core_priorities(self):
        """Initialize core priorities based on type."""
        for core in self.cores:
            if core.core_type == CoreType.PERFORMANCE:
                core.thread_affinity = self.config.performance_core_priority
            else:
                core.thread_affinity = self.config.efficiency_core_priority
    
    def get_optimal_cores(self, workload_profile: WorkloadProfile, 
                         num_cores: int = 1) -> List[CoreInfo]:
        """Get optimal cores for a workload."""
        # Filter available cores
        available_cores = [core for core in self.cores if core.is_available]
        
        # Sort by suitability for workload
        if workload_profile.preferred_core_type == CoreType.PERFORMANCE:
            suitable_cores = [core for core in available_cores 
                            if core.core_type == CoreType.PERFORMANCE]
            fallback_cores = [core for core in available_cores 
                            if core.core_type == CoreType.EFFICIENCY]
        else:
            suitable_cores = [core for core in available_cores 
                            if core.core_type == CoreType.EFFICIENCY]
            fallback_cores = [core for core in available_cores 
                            if core.core_type == CoreType.PERFORMANCE]
        
        # Sort by utilization (least utilized first)
        suitable_cores.sort(key=lambda c: self.core_utilization[c.core_id])
        fallback_cores.sort(key=lambda c: self.core_utilization[c.core_id])
        
        # Select cores
        selected_cores = suitable_cores[:num_cores]
        if len(selected_cores) < num_cores:
            needed = num_cores - len(selected_cores)
            selected_cores.extend(fallback_cores[:needed])
        
        return selected_cores[:num_cores]
    
    def assign_thread_to_core(self, thread_id: int, core: CoreInfo) -> bool:
        """Assign a thread to a specific core."""
        try:
            if self.config.enable_core_affinity:
                # Set thread affinity (simplified - real implementation would use proper system calls)
                os.sched_setaffinity(0, {core.core_id})
            
            self.thread_assignments[thread_id] = core.core_id
            self.core_utilization[core.core_id] += 0.1  # Estimate 10% utilization per thread
            
            self.logger.debug(f"🧵 Assigned thread {thread_id} to core {core.core_id}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to assign thread {thread_id} to core {core.core_id}: {e}")
            return False
    
    def release_thread_from_core(self, thread_id: int):
        """Release a thread from its assigned core."""
        if thread_id in self.thread_assignments:
            core_id = self.thread_assignments[thread_id]
            self.core_utilization[core_id] = max(0, self.core_utilization[core_id] - 0.1)
            del self.thread_assignments[thread_id]
    
    def get_core_utilization(self) -> Dict[int, float]:
        """Get current core utilization."""
        return self.core_utilization.copy()
    
    def get_available_cores(self) -> List[CoreInfo]:
        """Get available cores."""
        return [core for core in self.cores if core.is_available]

class WorkloadBalancer:
    """Balances workloads across CPU cores."""
    
    def __init__(self, config: CPUConfig, core_manager: CoreManager):
        self.config = config
        self.core_manager = core_manager
        self.logger = logger.getChild('WorkloadBalancer')
        
        # Workload tracking
        self.active_workloads = {}
        self.workload_queue = queue.PriorityQueue()
        self.balance_history = []
        
        # Start balancing thread
        if self.config.enable_workload_balancing:
            self._start_balancing()
    
    def _start_balancing(self):
        """Start workload balancing thread."""
        def balance():
            while True:
                try:
                    self._balance_workloads()
                    time.sleep(self.config.balance_check_interval)
                except Exception as e:
                    self.logger.error(f"Workload balancing error: {e}")
                    time.sleep(5)
        
        balance_thread = threading.Thread(target=balance, daemon=True)
        balance_thread.start()
        self.logger.info("⚖️ Workload balancing started")
    
    def _balance_workloads(self):
        """Balance workloads across cores."""
        if not self.active_workloads:
            return
        
        # Get current utilization
        utilization = self.core_manager.get_core_utilization()
        
        # Check for imbalance
        util_values = list(utilization.values())
        if not util_values:
            return
        
        max_util = max(util_values)
        min_util = min(util_values)
        imbalance = max_util - min_util
        
        if imbalance > self.config.load_balance_threshold:
            self.logger.info(f"⚖️ Load imbalance detected: {imbalance:.2f}")
            self._redistribute_workloads()
    
    def _redistribute_workloads(self):
        """Redistribute workloads for better balance."""
        # This is a simplified implementation
        # Real implementation would move threads between cores
        self.logger.debug("🔄 Redistributing workloads")
    
    def add_workload(self, workload_id: str, profile: WorkloadProfile):
        """Add a workload to the balancer."""
        self.active_workloads[workload_id] = {
            'profile': profile,
            'start_time': time.time(),
            'assigned_cores': []
        }
        
        # Get optimal cores
        num_cores = min(profile.priority, self.config.max_threads_per_core)
        optimal_cores = self.core_manager.get_optimal_cores(profile, num_cores)
        
        # Assign cores
        for core in optimal_cores:
            self.active_workloads[workload_id]['assigned_cores'].append(core.core_id)
        
        self.logger.debug(f"📋 Added workload {workload_id} with {len(optimal_cores)} cores")
    
    def remove_workload(self, workload_id: str):
        """Remove a workload from the balancer."""
        if workload_id in self.active_workloads:
            del self.active_workloads[workload_id]
            self.logger.debug(f"🗑️ Removed workload {workload_id}")

class M1AdvancedCPUOptimizer:
    """Advanced CPU optimizer for M1/M2/M3/M4 chips."""
    
    def __init__(self, config: Optional[CPUConfig] = None):
        self.config = config or CPUConfig()
        self.logger = logger.getChild('M1AdvancedCPUOptimizer')
        
        # Initialize components
        self.thermal_manager = ThermalManager(self.config)
        self.core_manager = CoreManager(self.config)
        self.workload_balancer = WorkloadBalancer(self.config, self.core_manager)
        
        # Thread pool
        self.thread_pool = None
        if self.config.enable_thread_pooling:
            self._initialize_thread_pool()
        
        # Performance tracking
        self.performance_metrics = {
            'total_operations': 0,
            'thermal_throttling_events': 0,
            'core_switches': 0,
            'workload_balances': 0,
            'average_execution_time': 0.0
        }
        
        self.logger.info("🚀 M1 Advanced CPU Optimizer initialized")
    
    def _initialize_thread_pool(self):
        """Initialize thread pool for parallel execution."""
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.thread_pool_size,
            thread_name_prefix="M1CPU"
        )
        self.logger.info(f"🧵 Thread pool initialized with {self.config.thread_pool_size} workers")
    
    def optimize_for_workload(self, workload_type: WorkloadType, 
                            intensity: float = 0.5) -> WorkloadProfile:
        """Create optimized workload profile."""
        profiles = {
            WorkloadType.CPU_INTENSIVE: WorkloadProfile(
                workload_type=workload_type,
                cpu_intensity=0.9,
                memory_intensity=0.3,
                io_intensity=0.1,
                preferred_core_type=CoreType.PERFORMANCE,
                estimated_duration=60.0,
                priority=8,
                parallelizable=True
            ),
            WorkloadType.MEMORY_INTENSIVE: WorkloadProfile(
                workload_type=workload_type,
                cpu_intensity=0.4,
                memory_intensity=0.9,
                io_intensity=0.2,
                preferred_core_type=CoreType.UNIFIED,
                estimated_duration=30.0,
                priority=6,
                parallelizable=True
            ),
            WorkloadType.IO_INTENSIVE: WorkloadProfile(
                workload_type=workload_type,
                cpu_intensity=0.2,
                memory_intensity=0.3,
                io_intensity=0.9,
                preferred_core_type=CoreType.EFFICIENCY,
                estimated_duration=120.0,
                priority=4,
                parallelizable=False
            ),
            WorkloadType.MIXED: WorkloadProfile(
                workload_type=workload_type,
                cpu_intensity=0.6,
                memory_intensity=0.6,
                io_intensity=0.4,
                preferred_core_type=CoreType.UNIFIED,
                estimated_duration=90.0,
                priority=5,
                parallelizable=True
            ),
            WorkloadType.BATCH: WorkloadProfile(
                workload_type=workload_type,
                cpu_intensity=0.8,
                memory_intensity=0.7,
                io_intensity=0.3,
                preferred_core_type=CoreType.PERFORMANCE,
                estimated_duration=300.0,
                priority=7,
                parallelizable=True
            ),
            WorkloadType.STREAMING: WorkloadProfile(
                workload_type=workload_type,
                cpu_intensity=0.5,
                memory_intensity=0.4,
                io_intensity=0.8,
                preferred_core_type=CoreType.EFFICIENCY,
                estimated_duration=float('inf'),
                priority=3,
                parallelizable=False
            )
        }
        
        profile = profiles.get(workload_type, profiles[WorkloadType.MIXED])
        
        # Adjust intensity
        profile.cpu_intensity *= intensity
        profile.memory_intensity *= intensity
        profile.io_intensity *= intensity
        
        return profile
    
    def execute_with_optimization(self, func: Callable, *args, 
                                 workload_type: WorkloadType = WorkloadType.MIXED,
                                 **kwargs) -> Any:
        """Execute function with CPU optimization."""
        # Create workload profile
        profile = self.optimize_for_workload(workload_type)
        
        # Check thermal state
        if self.thermal_manager.is_throttling_active():
            self.logger.warning("🌡️ CPU throttling active - performance may be reduced")
            self.performance_metrics['thermal_throttling_events'] += 1
        
        # Get optimal cores
        optimal_cores = self.core_manager.get_optimal_cores(profile, 1)
        
        if not optimal_cores:
            self.logger.warning("⚠️ No available cores - using default execution")
            return func(*args, **kwargs)
        
        # Assign to core
        thread_id = threading.get_ident()
        core = optimal_cores[0]
        self.core_manager.assign_thread_to_core(thread_id, core)
        
        try:
            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Update metrics
            self.performance_metrics['total_operations'] += 1
            self.performance_metrics['average_execution_time'] = (
                (self.performance_metrics['average_execution_time'] * 
                 (self.performance_metrics['total_operations'] - 1) + execution_time) /
                self.performance_metrics['total_operations']
            )
            
            self.logger.debug(f"⚡ Executed on core {core.core_id} in {execution_time:.3f}s")
            
            return result
            
        finally:
            # Release core
            self.core_manager.release_thread_from_core(thread_id)
    
    def parallel_execute(self, func: Callable, data_list: List[Any], 
                        workload_type: WorkloadType = WorkloadType.MIXED,
                        max_workers: Optional[int] = None) -> List[Any]:
        """Execute function in parallel with CPU optimization."""
        if not self.thread_pool:
            self.logger.warning("⚠️ Thread pool not available - using sequential execution")
            return [func(item) for item in data_list]
        
        # Create workload profile
        profile = self.optimize_for_workload(workload_type)
        
        # Determine number of workers
        if max_workers is None:
            available_cores = self.core_manager.get_available_cores()
            max_workers = min(len(available_cores), len(data_list))
        
        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, item in enumerate(data_list):
                future = executor.submit(self.execute_with_optimization, func, item, workload_type)
                futures.append(future)
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Parallel execution error: {e}")
                    results.append(None)
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'cpu_metrics': self.performance_metrics,
            'thermal_state': self.thermal_manager.get_thermal_state().value,
            'core_utilization': self.core_manager.get_core_utilization(),
            'available_cores': len(self.core_manager.get_available_cores()),
            'active_workloads': len(self.workload_balancer.active_workloads)
        }
    
    def shutdown(self):
        """Shutdown optimizer."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        self.logger.info("🛑 M1 Advanced CPU Optimizer shutdown")

# Global instance
_advanced_cpu_optimizer: Optional[M1AdvancedCPUOptimizer] = None

def get_advanced_cpu_optimizer(config: Optional[CPUConfig] = None) -> M1AdvancedCPUOptimizer:
    """Get or create the global advanced CPU optimizer."""
    global _advanced_cpu_optimizer
    
    if _advanced_cpu_optimizer is None:
        _advanced_cpu_optimizer = M1AdvancedCPUOptimizer(config)
    
    return _advanced_cpu_optimizer

def optimize_cpu_execution(workload_type: WorkloadType = WorkloadType.MIXED):
    """Decorator to optimize CPU execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_advanced_cpu_optimizer()
            return optimizer.execute_with_optimization(func, *args, workload_type=workload_type, **kwargs)
        return wrapper
    return decorator

def parallel_cpu_execution(workload_type: WorkloadType = WorkloadType.MIXED, max_workers: Optional[int] = None):
    """Decorator for parallel CPU execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(data_list, *args, **kwargs):
            optimizer = get_advanced_cpu_optimizer()
            return optimizer.parallel_execute(func, data_list, workload_type=workload_type, max_workers=max_workers)
        return wrapper
    return decorator

def get_cpu_performance_metrics() -> Dict[str, Any]:
    """Get CPU performance metrics."""
    optimizer = get_advanced_cpu_optimizer()
    return optimizer.get_performance_metrics()