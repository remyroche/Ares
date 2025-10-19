"""
M1-Optimized Parallel Processor

Implements advanced parallel processing with M1's performance/efficiency core
distinction, adaptive resource allocation, and intelligent task scheduling.
"""

import multiprocessing
import concurrent.futures
import threading
import time
import os
import asyncio
import psutil
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass
import logging
import queue
from enum import Enum

from src.utils.tprint import tprint

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Types of tasks for optimal core assignment."""
    CPU_INTENSIVE = "cpu_intensive"
    IO_BOUND = "io_bound"
    MEMORY_INTENSIVE = "memory_intensive"
    MIXED = "mixed"

@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    
    # Core allocation
    performance_cores: int = 4  # M1 performance cores
    efficiency_cores: int = 4   # M1 efficiency cores
    max_workers: int = 8
    
    # Task scheduling
    cpu_intensive_ratio: float = 0.6  # 60% to performance cores
    io_bound_ratio: float = 0.4       # 40% to efficiency cores
    
    # Adaptive settings
    enable_adaptive_allocation: bool = True
    adaptation_interval: float = 30.0  # seconds
    cpu_threshold: float = 80.0       # CPU usage threshold
    memory_threshold: float = 85.0    # Memory usage threshold
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval: float = 5.0  # seconds

class TaskScheduler:
    """Intelligent task scheduler for M1 cores."""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = logger.getChild('TaskScheduler')
        
        # Core assignments
        self.performance_queue = queue.Queue()
        self.efficiency_queue = queue.Queue()
        
        # Performance tracking
        self.task_history = []
        self.core_utilization = {'performance': 0.0, 'efficiency': 0.0}
        
        # Thread pools
        self.performance_executor = None
        self.efficiency_executor = None
        
        self._initialize_executors()
    
    def _initialize_executors(self):
        """Initialize thread pools for different core types."""
        performance_workers = int(self.config.max_workers * self.config.cpu_intensive_ratio)
        efficiency_workers = int(self.config.max_workers * self.config.io_bound_ratio)
        
        self.performance_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=performance_workers,
            thread_name_prefix="PerformanceCore"
        )
        
        self.efficiency_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=efficiency_workers,
            thread_name_prefix="EfficiencyCore"
        )
        
        tprint(f"🚀 [PARALLEL] Initialized executors: {performance_workers} performance, {efficiency_workers} efficiency cores")
    
    def schedule_task(self, task: Callable, task_type: TaskType, *args, **kwargs) -> concurrent.futures.Future:
        """Schedule a task on the appropriate core type."""
        if task_type == TaskType.CPU_INTENSIVE:
            executor = self.performance_executor
            queue_name = "performance"
        else:
            executor = self.efficiency_executor
            queue_name = "efficiency"
        
        future = executor.submit(task, *args, **kwargs)
        
        # Track task
        self.task_history.append({
            'task_type': task_type.value,
            'queue': queue_name,
            'timestamp': time.time(),
            'future': future
        })
        
        return future
    
    def get_optimal_task_type(self, task_characteristics: Dict[str, Any]) -> TaskType:
        """Determine optimal task type based on characteristics."""
        cpu_usage = task_characteristics.get('cpu_usage', 0.5)
        io_operations = task_characteristics.get('io_operations', 0)
        memory_usage = task_characteristics.get('memory_usage', 0.1)
        
        if cpu_usage > 0.7 and io_operations < 0.3:
            return TaskType.CPU_INTENSIVE
        elif io_operations > 0.5:
            return TaskType.IO_BOUND
        elif memory_usage > 0.5:
            return TaskType.MEMORY_INTENSIVE
        else:
            return TaskType.MIXED

class M1ParallelProcessor:
    """M1-optimized parallel processor with intelligent core allocation."""
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        self.config = config or ParallelConfig()
        self.logger = logger.getChild('M1ParallelProcessor')
        
        # M1 detection
        self.is_m1 = self._detect_m1_system()
        self.m1_generation = self._detect_m1_generation()
        
        # Initialize components
        self.task_scheduler = TaskScheduler(self.config)
        self.performance_monitor = None
        
        # Resource tracking
        self.system_stats = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'last_update': time.time()
        }
        
        # Adaptive settings
        self.adaptive_workers = self.config.max_workers
        self.last_adaptation = time.time()
        
        if self.config.enable_performance_monitoring:
            self._start_performance_monitoring()
        
        tprint("🚀 [PARALLEL] M1 Parallel Processor initialized")
        tprint(f"📊 [PARALLEL] M1 System: {self.is_m1}, Generation: {self.m1_generation}")
    
    def _detect_m1_system(self) -> bool:
        """Detect if running on M1 system."""
        try:
            import platform
            import subprocess
            
            if platform.system() != 'Darwin':
                return False
            
            result = subprocess.run(['sysctl', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                brand = result.stdout.strip().lower()
                m1_indicators = ['apple', 'm1', 'm2', 'm3', 'm4', 'silicon']
                return any(indicator in brand for indicator in m1_indicators)
            
            return False
        except Exception as e:
            self.logger.warning(f"Could not detect M1 hardware: {e}")
            return False
    
    def _detect_m1_generation(self) -> str:
        """Detect M1 chip generation."""
        try:
            import subprocess
            
            result = subprocess.run(['sysctl', 'hw.model'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                model = result.stdout.strip().lower()
                if 'm1' in model:
                    return 'M1'
                elif 'm2' in model:
                    return 'M2'
                elif 'm3' in model:
                    return 'M3'
                elif 'm4' in model:
                    return 'M4'
            
            return 'Unknown'
        except Exception:
            return 'Unknown'
    
    def _start_performance_monitoring(self):
        """Start performance monitoring thread."""
        def monitor_performance():
            while True:
                try:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent
                    
                    self.system_stats.update({
                        'cpu_usage': cpu_percent,
                        'memory_usage': memory_percent,
                        'last_update': time.time()
                    })
                    
                    # Adaptive resource allocation
                    if self.config.enable_adaptive_allocation:
                        self._adaptive_resource_allocation()
                    
                    time.sleep(self.config.monitoring_interval)
                    
                except Exception as e:
                    self.logger.error(f"Performance monitoring error: {e}")
                    time.sleep(self.config.monitoring_interval)
        
        monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
        monitor_thread.start()
        self.performance_monitor = monitor_thread
        
        tprint("📊 [PARALLEL] Performance monitoring started")
    
    def _adaptive_resource_allocation(self):
        """Dynamically adjust resources based on system state."""
        current_time = time.time()
        
        # Only adapt if enough time has passed
        if current_time - self.last_adaptation < self.config.adaptation_interval:
            return
        
        cpu_usage = self.system_stats['cpu_usage']
        memory_usage = self.system_stats['memory_usage']
        
        old_workers = self.adaptive_workers
        
        # Adjust based on CPU usage
        if cpu_usage > self.config.cpu_threshold:
            # Reduce parallelism to avoid context switching overhead
            self.adaptive_workers = max(2, self.adaptive_workers - 1)
            tprint(f"📉 [PARALLEL] Reduced workers due to high CPU usage: {old_workers} -> {self.adaptive_workers}")
        elif cpu_usage < 50:
            # Increase parallelism
            self.adaptive_workers = min(self.config.max_workers, self.adaptive_workers + 1)
            tprint(f"📈 [PARALLEL] Increased workers due to low CPU usage: {old_workers} -> {self.adaptive_workers}")
        
        # Adjust based on memory usage
        if memory_usage > self.config.memory_threshold:
            # Reduce parallelism to conserve memory
            self.adaptive_workers = max(2, self.adaptive_workers - 1)
            tprint(f"📉 [PARALLEL] Reduced workers due to high memory usage: {old_workers} -> {self.adaptive_workers}")
        
        self.last_adaptation = current_time
    
    def parallel_map(self, func: Callable, data_list: List[Any], 
                    task_type: TaskType = TaskType.MIXED,
                    **kwargs) -> List[Any]:
        """Parallel map with intelligent task scheduling."""
        tprint(f"🔄 [PARALLEL] Parallel map: {len(data_list)} tasks, type: {task_type.value}")
        
        # Determine optimal task type if not specified
        if task_type == TaskType.MIXED:
            task_characteristics = {
                'cpu_usage': 0.5,  # Default assumption
                'io_operations': 0.1,
                'memory_usage': 0.1
            }
            task_type = self.task_scheduler.get_optimal_task_type(task_characteristics)
        
        # Schedule tasks
        futures = []
        for item in data_list:
            future = self.task_scheduler.schedule_task(func, task_type, item, **kwargs)
            futures.append(future)
        
        # Collect results
        results = []
        completed = 0
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % max(1, len(futures) // 10) == 0:  # Progress every 10%
                    progress = completed / len(futures) * 100
                    tprint(f"📊 [PARALLEL] Progress: {progress:.1f}% ({completed}/{len(futures)})")
                    
            except Exception as e:
                tprint(f"❌ [PARALLEL] Task failed: {e}")
                results.append(None)
        
        tprint(f"✅ [PARALLEL] Parallel map completed: {len(results)} results")
        return results
    
    def parallel_apply(self, func: Callable, data: pd.DataFrame,
                      axis: int = 0, task_type: TaskType = TaskType.MIXED,
                      **kwargs) -> pd.DataFrame:
        """Parallel apply for DataFrames with chunking."""
        tprint(f"🔄 [PARALLEL] Parallel apply: {data.shape}, axis: {axis}")
        
        # Determine chunk size based on available workers
        chunk_size = max(1, len(data) // self.adaptive_workers)
        
        # Create chunks
        if axis == 0:  # Apply to rows
            chunks = [data.iloc[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        else:  # Apply to columns
            chunks = [data.iloc[:, i:i+chunk_size] for i in range(0, len(data.columns), chunk_size)]
        
        tprint(f"📊 [PARALLEL] Created {len(chunks)} chunks for parallel processing")
        
        # Process chunks in parallel
        results = self.parallel_map(func, chunks, task_type, **kwargs)
        
        # Combine results
        if axis == 0:
            combined_result = pd.concat(results, ignore_index=True)
        else:
            combined_result = pd.concat(results, axis=1)
        
        tprint(f"✅ [PARALLEL] Parallel apply completed: {combined_result.shape}")
        return combined_result
    
    def cpu_intensive_parallel_map(self, func: Callable, data_list: List[Any], **kwargs) -> List[Any]:
        """Parallel map optimized for CPU-intensive tasks."""
        return self.parallel_map(func, data_list, TaskType.CPU_INTENSIVE, **kwargs)
    
    def io_bound_parallel_map(self, func: Callable, data_list: List[Any], **kwargs) -> List[Any]:
        """Parallel map optimized for I/O-bound tasks."""
        return self.parallel_map(func, data_list, TaskType.IO_BOUND, **kwargs)
    
    def memory_intensive_parallel_map(self, func: Callable, data_list: List[Any], **kwargs) -> List[Any]:
        """Parallel map optimized for memory-intensive tasks."""
        return self.parallel_map(func, data_list, TaskType.MEMORY_INTENSIVE, **kwargs)
    
    def get_optimal_thread_count(self) -> int:
        """Get optimal thread count based on current system state."""
        cpu_usage = self.system_stats['cpu_usage']
        memory_usage = self.system_stats['memory_usage']
        
        # Base thread count
        base_threads = self.config.max_workers
        
        # Adjust based on system state
        if cpu_usage > 80:
            return max(2, base_threads // 2)
        elif cpu_usage < 30:
            return min(self.config.max_workers * 2, base_threads)
        elif memory_usage > 85:
            return max(2, base_threads // 2)
        else:
            return base_threads
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'is_m1': self.is_m1,
            'm1_generation': self.m1_generation,
            'adaptive_workers': self.adaptive_workers,
            'system_stats': self.system_stats.copy(),
            'task_history_count': len(self.task_scheduler.task_history),
            'core_utilization': self.task_scheduler.core_utilization.copy()
        }
        
        return stats
    
    def cleanup(self):
        """Clean up parallel processing resources."""
        tprint("🧹 [PARALLEL] Cleaning up parallel processor")
        
        # Shutdown executors
        if self.task_scheduler.performance_executor:
            self.task_scheduler.performance_executor.shutdown(wait=True)
        
        if self.task_scheduler.efficiency_executor:
            self.task_scheduler.efficiency_executor.shutdown(wait=True)
        
        # Clear task history
        self.task_scheduler.task_history.clear()
        
        tprint("✅ [PARALLEL] Parallel processor cleanup completed")
