"""
Multi-Model Orchestrator for A/B/C Testing

This module provides a sophisticated orchestrator to manage and coordinate
multiple model testing with advanced scheduling, resource management, and
performance optimization.

Key Features:
- Intelligent model scheduling and load balancing
- Resource management and optimization
- Model lifecycle management
- Performance monitoring and auto-scaling
- Fault tolerance and error recovery
- Advanced coordination algorithms
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json
import threading
from queue import Queue, PriorityQueue
import uuid

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.performance_utils import PerformanceMonitor
from src.utils.monitoring_utils import SystemMonitor

# Model management
from src.utils.standardized_model_manager import StandardizedModelManager
from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelConfig, ModelType

# Core decorators and validation
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

logger = logging.getLogger(__name__)


class ModelPriority(Enum):
    """Model execution priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class ExecutionStrategy(Enum):
    """Model execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    ADAPTIVE = "adaptive"
    LOAD_BALANCED = "load_balanced"


class ResourceType(Enum):
    """Resource types for allocation."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class ResourceAllocation:
    """Resource allocation specification."""
    resource_type: ResourceType
    allocated: float
    maximum: float
    unit: str = "percentage"
    priority: int = 1


@dataclass
class ModelTask:
    """Model execution task."""
    task_id: str
    model_id: str
    model_name: str
    priority: ModelPriority
    estimated_duration: float
    resource_requirements: Dict[ResourceType, float]
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class OrchestratorConfig:
    """Configuration for the multi-model orchestrator."""
    # Basic configuration
    max_concurrent_models: int = 4
    max_workers: int = 8
    execution_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    
    # Resource management
    cpu_limit: float = 80.0  # percentage
    memory_limit: float = 80.0  # percentage
    gpu_limit: float = 90.0  # percentage
    disk_limit: float = 85.0  # percentage
    
    # Performance settings
    enable_auto_scaling: bool = True
    enable_load_balancing: bool = True
    enable_fault_tolerance: bool = True
    max_retries: int = 3
    retry_delay: float = 5.0
    
    # Monitoring settings
    monitoring_interval: int = 30  # seconds
    performance_threshold: float = 0.8
    enable_alerting: bool = True
    
    # Scheduling settings
    enable_priority_scheduling: bool = True
    enable_dependency_resolution: bool = True
    max_queue_size: int = 1000
    
    # Optimization settings
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_gpu_optimization: bool = True


class ResourceManager:
    """Advanced resource management system."""
    
    def __init__(self, config: OrchestratorConfig):
        """Initialize resource manager."""
        self.config = config
        self.logger = logger.getChild('ResourceManager')
        
        # Resource tracking
        self.resource_usage: Dict[ResourceType, float] = {
            ResourceType.CPU: 0.0,
            ResourceType.MEMORY: 0.0,
            ResourceType.GPU: 0.0,
            ResourceType.DISK: 0.0,
            ResourceType.NETWORK: 0.0
        }
        
        # Resource limits
        self.resource_limits: Dict[ResourceType, float] = {
            ResourceType.CPU: config.cpu_limit,
            ResourceType.MEMORY: config.memory_limit,
            ResourceType.GPU: config.gpu_limit,
            ResourceType.DISK: config.disk_limit,
            ResourceType.NETWORK: 100.0  # Default network limit
        }
        
        # Active allocations
        self.active_allocations: Dict[str, Dict[ResourceType, float]] = {}
        
        self.logger.info("🚀 ResourceManager initialized")
        self.logger.info(f"📊 Resource limits: CPU={config.cpu_limit}%, Memory={config.memory_limit}%, GPU={config.gpu_limit}%")
    
    def can_allocate_resources(self, requirements: Dict[ResourceType, float]) -> bool:
        """Check if resources can be allocated."""
        for resource_type, required in requirements.items():
            current_usage = self.resource_usage.get(resource_type, 0.0)
            limit = self.resource_limits.get(resource_type, 100.0)
            
            if current_usage + required > limit:
                self.logger.debug(f"❌ Cannot allocate {required}% of {resource_type.value}: {current_usage}% + {required}% > {limit}%")
                return False
        
        return True
    
    def allocate_resources(self, task_id: str, requirements: Dict[ResourceType, float]) -> bool:
        """Allocate resources for a task."""
        if not self.can_allocate_resources(requirements):
            return False
        
        # Allocate resources
        for resource_type, required in requirements.items():
            self.resource_usage[resource_type] += required
        
        # Track allocation
        self.active_allocations[task_id] = requirements.copy()
        
        self.logger.debug(f"✅ Allocated resources for {task_id}: {requirements}")
        return True
    
    def deallocate_resources(self, task_id: str) -> None:
        """Deallocate resources for a task."""
        if task_id not in self.active_allocations:
            return
        
        requirements = self.active_allocations[task_id]
        
        # Deallocate resources
        for resource_type, required in requirements.items():
            self.resource_usage[resource_type] -= required
            self.resource_usage[resource_type] = max(0.0, self.resource_usage[resource_type])
        
        # Remove allocation tracking
        del self.active_allocations[task_id]
        
        self.logger.debug(f"🗑️ Deallocated resources for {task_id}")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        return {
            'usage': {k.value: v for k, v in self.resource_usage.items()},
            'limits': {k.value: v for k, v in self.resource_limits.items()},
            'available': {
                k.value: v - self.resource_usage.get(k, 0.0) 
                for k, v in self.resource_limits.items()
            },
            'active_allocations': len(self.active_allocations)
        }
    
    def update_system_resources(self) -> None:
        """Update resource usage from system."""
        try:
            # CPU usage
            self.resource_usage[ResourceType.CPU] = psutil.cpu_percent()
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.resource_usage[ResourceType.MEMORY] = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.resource_usage[ResourceType.DISK] = (disk.used / disk.total) * 100
            
            # GPU usage (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.resource_usage[ResourceType.GPU] = gpus[0].load * 100
            except ImportError:
                pass  # GPU monitoring not available
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not update system resources: {e}")


class TaskScheduler:
    """Advanced task scheduling system."""
    
    def __init__(self, config: OrchestratorConfig):
        """Initialize task scheduler."""
        self.config = config
        self.logger = logger.getChild('TaskScheduler')
        
        # Task queues by priority
        self.task_queues: Dict[ModelPriority, PriorityQueue] = {
            priority: PriorityQueue(maxsize=config.max_queue_size)
            for priority in ModelPriority
        }
        
        # Task tracking
        self.pending_tasks: Dict[str, ModelTask] = {}
        self.running_tasks: Dict[str, ModelTask] = {}
        self.completed_tasks: Dict[str, ModelTask] = {}
        self.failed_tasks: Dict[str, ModelTask] = {}
        
        # Dependency tracking
        self.task_dependencies: Dict[str, List[str]] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        
        self.logger.info("🚀 TaskScheduler initialized")
        self.logger.info(f"📊 Max concurrent models: {config.max_concurrent_models}")
    
    def add_task(self, task: ModelTask) -> bool:
        """Add a task to the scheduler."""
        try:
            # Validate task
            if not self._validate_task(task):
                return False
            
            # Add to appropriate queue
            priority_score = task.priority.value
            self.task_queues[task.priority].put((priority_score, task.task_id, task))
            
            # Track task
            self.pending_tasks[task.task_id] = task
            
            # Build dependency graph
            if task.dependencies:
                self.task_dependencies[task.task_id] = task.dependencies
                for dep in task.dependencies:
                    if dep not in self.dependency_graph:
                        self.dependency_graph[dep] = []
                    self.dependency_graph[dep].append(task.task_id)
            
            self.logger.debug(f"✅ Added task {task.task_id} with priority {task.priority.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add task {task.task_id}: {e}")
            return False
    
    def get_next_task(self) -> Optional[ModelTask]:
        """Get the next task to execute."""
        # Check tasks in priority order
        for priority in ModelPriority:
            queue = self.task_queues[priority]
            
            if not queue.empty():
                try:
                    # Get task from queue
                    priority_score, task_id, task = queue.get_nowait()
                    
                    # Check if dependencies are satisfied
                    if self._are_dependencies_satisfied(task):
                        return task
                    else:
                        # Put task back in queue
                        queue.put((priority_score, task_id, task))
                        
                except:
                    continue  # Queue is empty
        
        return None
    
    def start_task(self, task: ModelTask) -> None:
        """Mark a task as started."""
        task.started_at = datetime.now()
        task.status = "running"
        
        # Move from pending to running
        if task.task_id in self.pending_tasks:
            del self.pending_tasks[task.task_id]
        self.running_tasks[task.task_id] = task
        
        self.logger.debug(f"🚀 Started task {task.task_id}")
    
    def complete_task(self, task: ModelTask, result: Any = None) -> None:
        """Mark a task as completed."""
        task.completed_at = datetime.now()
        task.status = "completed"
        task.result = result
        
        # Move from running to completed
        if task.task_id in self.running_tasks:
            del self.running_tasks[task.task_id]
        self.completed_tasks[task.task_id] = task
        
        self.logger.debug(f"✅ Completed task {task.task_id}")
    
    def fail_task(self, task: ModelTask, error: str) -> None:
        """Mark a task as failed."""
        task.completed_at = datetime.now()
        task.status = "failed"
        task.error = error
        
        # Move from running to failed
        if task.task_id in self.running_tasks:
            del self.running_tasks[task.task_id]
        self.failed_tasks[task.task_id] = task
        
        self.logger.error(f"❌ Failed task {task.task_id}: {error}")
    
    def _validate_task(self, task: ModelTask) -> bool:
        """Validate a task before adding."""
        if not task.task_id or not task.model_id:
            return False
        
        if task.task_id in self.pending_tasks or task.task_id in self.running_tasks:
            self.logger.warning(f"⚠️ Task {task.task_id} already exists")
            return False
        
        return True
    
    def _are_dependencies_satisfied(self, task: ModelTask) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        return True
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            'pending_tasks': len(self.pending_tasks),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'queue_sizes': {
                priority.name: queue.qsize() 
                for priority, queue in self.task_queues.items()
            }
        }


class ModelExecutor:
    """Model execution engine with advanced capabilities."""
    
    def __init__(self, config: OrchestratorConfig):
        """Initialize model executor."""
        self.config = config
        self.logger = logger.getChild('ModelExecutor')
        
        # Execution pools
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(config.max_workers, 4))
        
        # Model instances
        self.model_instances: Dict[str, Any] = {}
        self.model_factory = EnhancedModelFactory()
        
        # Performance tracking
        self.execution_stats: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("🚀 ModelExecutor initialized")
        self.logger.info(f"📊 Max workers: {config.max_workers}")
    
    async def execute_model_task(self, task: ModelTask, market_data: pd.DataFrame) -> Any:
        """Execute a model task."""
        start_time = time.time()
        
        try:
            self.logger.info(f"🔄 Executing model task {task.task_id} for {task.model_name}")
            
            # Get or create model instance
            model = await self._get_model_instance(task.model_id, task.model_name)
            
            # Execute model based on strategy
            if self.config.execution_strategy == ExecutionStrategy.SEQUENTIAL:
                result = await self._execute_sequential(model, market_data, task)
            elif self.config.execution_strategy == ExecutionStrategy.PARALLEL:
                result = await self._execute_parallel(model, market_data, task)
            elif self.config.execution_strategy == ExecutionStrategy.PIPELINE:
                result = await self._execute_pipeline(model, market_data, task)
            else:
                result = await self._execute_adaptive(model, market_data, task)
            
            # Update execution stats
            execution_time = time.time() - start_time
            self._update_execution_stats(task.model_id, execution_time, True)
            
            self.logger.info(f"✅ Model task {task.task_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_execution_stats(task.model_id, execution_time, False)
            
            self.logger.error(f"❌ Model task {task.task_id} failed after {execution_time:.2f}s: {e}")
            raise
    
    async def _get_model_instance(self, model_id: str, model_name: str) -> Any:
        """Get or create model instance."""
        if model_id in self.model_instances:
            return self.model_instances[model_id]
        
        # Create new model instance
        # This is a placeholder - in practice, you would load the actual model
        model = await self._create_model_instance(model_name)
        self.model_instances[model_id] = model
        
        return model
    
    async def _create_model_instance(self, model_name: str) -> Any:
        """Create a new model instance."""
        # This is a placeholder implementation
        # In practice, you would use the model factory to create actual models
        
        class PlaceholderModel:
            def __init__(self, name: str):
                self.name = name
                self.is_fitted = False
            
            async def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
                # Placeholder prediction
                return {
                    'action': 'hold',
                    'size': 0.0,
                    'confidence': 0.5,
                    'model_name': self.name
                }
        
        return PlaceholderModel(model_name)
    
    async def _execute_sequential(self, model: Any, market_data: pd.DataFrame, task: ModelTask) -> Any:
        """Execute model sequentially."""
        results = []
        
        for timestamp, bar in market_data.iterrows():
            result = await model.predict(bar.to_frame().T)
            results.append({
                'timestamp': timestamp,
                'result': result
            })
        
        return results
    
    async def _execute_parallel(self, model: Any, market_data: pd.DataFrame, task: ModelTask) -> Any:
        """Execute model in parallel."""
        # Split data into chunks
        chunk_size = max(1, len(market_data) // self.config.max_workers)
        chunks = [market_data.iloc[i:i+chunk_size] for i in range(0, len(market_data), chunk_size)]
        
        # Execute chunks in parallel
        tasks = []
        for chunk in chunks:
            task_future = asyncio.create_task(self._process_chunk(model, chunk))
            tasks.append(task_future)
        
        # Collect results
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        flattened_results = []
        for chunk_results in results:
            flattened_results.extend(chunk_results)
        
        return flattened_results
    
    async def _execute_pipeline(self, model: Any, market_data: pd.DataFrame, task: ModelTask) -> Any:
        """Execute model using pipeline processing."""
        # This would implement pipeline processing
        # For now, fall back to sequential
        return await self._execute_sequential(model, market_data, task)
    
    async def _execute_adaptive(self, model: Any, market_data: pd.DataFrame, task: ModelTask) -> Any:
        """Execute model using adaptive strategy."""
        # Choose strategy based on data size and system resources
        data_size = len(market_data)
        
        if data_size < 1000:
            return await self._execute_sequential(model, market_data, task)
        else:
            return await self._execute_parallel(model, market_data, task)
    
    async def _process_chunk(self, model: Any, chunk: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process a chunk of data."""
        results = []
        
        for timestamp, bar in chunk.iterrows():
            result = await model.predict(bar.to_frame().T)
            results.append({
                'timestamp': timestamp,
                'result': result
            })
        
        return results
    
    def _update_execution_stats(self, model_id: str, execution_time: float, success: bool) -> None:
        """Update execution statistics."""
        if model_id not in self.execution_stats:
            self.execution_stats[model_id] = {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'total_time': 0.0,
                'average_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0
            }
        
        stats = self.execution_stats[model_id]
        stats['total_executions'] += 1
        stats['total_time'] += execution_time
        stats['average_time'] = stats['total_time'] / stats['total_executions']
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
        
        if success:
            stats['successful_executions'] += 1
        else:
            stats['failed_executions'] += 1
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.execution_stats.copy()
    
    def cleanup(self) -> None:
        """Cleanup executor resources."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        self.model_instances.clear()


class MultiModelOrchestrator:
    """Advanced multi-model orchestrator for A/B/C testing."""
    
    def __init__(self, config: OrchestratorConfig):
        """Initialize the multi-model orchestrator."""
        self.config = config
        self.logger = logger.getChild('MultiModelOrchestrator')
        
        # Core components
        self.resource_manager = ResourceManager(config)
        self.task_scheduler = TaskScheduler(config)
        self.model_executor = ModelExecutor(config)
        
        # Monitoring
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        
        # State management
        self.is_running = False
        self.execution_loop_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.orchestrator_stats = {
            'total_tasks_processed': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0,
            'uptime': 0.0,
            'start_time': None
        }
        
        self.logger.info("🚀 MultiModelOrchestrator initialized")
        self.logger.info(f"📊 Execution strategy: {config.execution_strategy.value}")
        self.logger.info(f"📊 Max concurrent models: {config.max_concurrent_models}")
    
    async def start(self) -> None:
        """Start the orchestrator."""
        if self.is_running:
            self.logger.warning("⚠️ Orchestrator is already running")
            return
        
        self.logger.info("🚀 Starting MultiModelOrchestrator...")
        
        self.is_running = True
        self.orchestrator_stats['start_time'] = datetime.now()
        
        # Start execution loop
        self.execution_loop_task = asyncio.create_task(self._execution_loop())
        
        # Start monitoring
        if self.config.enable_alerting:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("✅ MultiModelOrchestrator started")
    
    async def stop(self) -> None:
        """Stop the orchestrator."""
        if not self.is_running:
            self.logger.warning("⚠️ Orchestrator is not running")
            return
        
        self.logger.info("🛑 Stopping MultiModelOrchestrator...")
        
        self.is_running = False
        
        # Cancel tasks
        if self.execution_loop_task:
            self.execution_loop_task.cancel()
            try:
                await self.execution_loop_task
            except asyncio.CancelledError:
                pass
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup
        self.model_executor.cleanup()
        
        # Update uptime
        if self.orchestrator_stats['start_time']:
            uptime = (datetime.now() - self.orchestrator_stats['start_time']).total_seconds()
            self.orchestrator_stats['uptime'] = uptime
        
        self.logger.info("✅ MultiModelOrchestrator stopped")
    
    async def submit_model_task(self, model_config: Dict[str, Any], 
                               priority: ModelPriority = ModelPriority.NORMAL,
                               dependencies: Optional[List[str]] = None) -> str:
        """Submit a model task for execution."""
        task_id = str(uuid.uuid4())
        
        # Create task
        task = ModelTask(
            task_id=task_id,
            model_id=model_config.get('model_id', f"model_{task_id}"),
            model_name=model_config.get('model_name', 'Unknown'),
            priority=priority,
            estimated_duration=model_config.get('estimated_duration', 60.0),
            resource_requirements=model_config.get('resource_requirements', {
                ResourceType.CPU: 20.0,
                ResourceType.MEMORY: 10.0
            }),
            dependencies=dependencies or []
        )
        
        # Add to scheduler
        if self.task_scheduler.add_task(task):
            self.logger.info(f"✅ Submitted model task {task_id} for {task.model_name}")
            return task_id
        else:
            self.logger.error(f"❌ Failed to submit model task {task_id}")
            raise ValueError(f"Failed to submit model task {task_id}")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check all task collections
        for task_dict in [
            self.task_scheduler.pending_tasks,
            self.task_scheduler.running_tasks,
            self.task_scheduler.completed_tasks,
            self.task_scheduler.failed_tasks
        ]:
            if task_id in task_dict:
                task = task_dict[task_id]
                return {
                    'task_id': task.task_id,
                    'model_id': task.model_id,
                    'model_name': task.model_name,
                    'status': task.status,
                    'priority': task.priority.name,
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'error': task.error
                }
        
        return None
    
    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        return {
            'is_running': self.is_running,
            'uptime': self.orchestrator_stats['uptime'],
            'statistics': self.orchestrator_stats,
            'scheduler_status': self.task_scheduler.get_scheduler_status(),
            'resource_status': self.resource_manager.get_resource_status(),
            'execution_stats': self.model_executor.get_execution_stats()
        }
    
    async def _execution_loop(self) -> None:
        """Main execution loop."""
        self.logger.info("🔄 Starting execution loop...")
        
        while self.is_running:
            try:
                # Update system resources
                self.resource_manager.update_system_resources()
                
                # Get next task
                task = self.task_scheduler.get_next_task()
                
                if task is None:
                    # No tasks available, wait a bit
                    await asyncio.sleep(1.0)
                    continue
                
                # Check if we can allocate resources
                if not self.resource_manager.can_allocate_resources(task.resource_requirements):
                    # Put task back in queue
                    self.task_scheduler.task_queues[task.priority].put((
                        task.priority.value, task.task_id, task
                    ))
                    await asyncio.sleep(1.0)
                    continue
                
                # Allocate resources
                if not self.resource_manager.allocate_resources(task.task_id, task.resource_requirements):
                    await asyncio.sleep(1.0)
                    continue
                
                # Start task
                self.task_scheduler.start_task(task)
                
                # Execute task
                try:
                    # Generate sample market data for testing
                    market_data = self._generate_sample_market_data()
                    
                    result = await self.model_executor.execute_model_task(task, market_data)
                    
                    # Complete task
                    self.task_scheduler.complete_task(task, result)
                    self.orchestrator_stats['successful_tasks'] += 1
                    
                except Exception as e:
                    # Fail task
                    self.task_scheduler.fail_task(task, str(e))
                    self.orchestrator_stats['failed_tasks'] += 1
                    
                    # Retry logic
                    if self.config.enable_fault_tolerance and task.task_id not in self.task_scheduler.failed_tasks:
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                
                finally:
                    # Always deallocate resources
                    self.resource_manager.deallocate_resources(task.task_id)
                
                # Update statistics
                self.orchestrator_stats['total_tasks_processed'] += 1
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in execution loop: {e}")
                await asyncio.sleep(1.0)
        
        self.logger.info("🛑 Execution loop stopped")
    
    async def _monitoring_loop(self) -> None:
        """Monitoring and alerting loop."""
        self.logger.info("📊 Starting monitoring loop...")
        
        while self.is_running:
            try:
                # Get system status
                status = await self.get_orchestrator_status()
                
                # Check performance thresholds
                if self._check_performance_thresholds(status):
                    await self._handle_performance_alert(status)
                
                # Update uptime
                if self.orchestrator_stats['start_time']:
                    uptime = (datetime.now() - self.orchestrator_stats['start_time']).total_seconds()
                    self.orchestrator_stats['uptime'] = uptime
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval)
        
        self.logger.info("🛑 Monitoring loop stopped")
    
    def _generate_sample_market_data(self) -> pd.DataFrame:
        """Generate sample market data for testing."""
        # Generate 100 data points
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        
        # Generate sample OHLCV data
        np.random.seed(42)
        base_price = 100.0
        returns = np.random.normal(0.0001, 0.02, 100)
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            volatility = abs(np.random.normal(0, 0.01))
            data.append({
                'timestamp': date,
                'open': prices[i-1] if i > 0 else price,
                'high': price * (1 + volatility),
                'low': price * (1 - volatility),
                'close': price,
                'volume': np.random.uniform(1000, 10000)
            })
        
        df = pd.DataFrame(data)
        df = df.set_index('timestamp')
        
        return df
    
    def _check_performance_thresholds(self, status: Dict[str, Any]) -> bool:
        """Check if performance thresholds are exceeded."""
        resource_status = status.get('resource_status', {})
        usage = resource_status.get('usage', {})
        
        # Check CPU usage
        if usage.get('cpu', 0) > self.config.performance_threshold * 100:
            return True
        
        # Check memory usage
        if usage.get('memory', 0) > self.config.performance_threshold * 100:
            return True
        
        return False
    
    async def _handle_performance_alert(self, status: Dict[str, Any]) -> None:
        """Handle performance alerts."""
        self.logger.warning("⚠️ Performance threshold exceeded!")
        self.logger.warning(f"📊 System status: {status}")
        
        # Implement auto-scaling or other mitigation strategies
        if self.config.enable_auto_scaling:
            await self._auto_scale()
    
    async def _auto_scale(self) -> None:
        """Implement auto-scaling logic."""
        self.logger.info("🔄 Implementing auto-scaling...")
        
        # This would implement actual auto-scaling logic
        # For now, just log the action
        self.logger.info("📈 Auto-scaling implemented")


# Convenience function for easy integration
async def create_orchestrator(config: Optional[OrchestratorConfig] = None) -> MultiModelOrchestrator:
    """Create and initialize a multi-model orchestrator."""
    if config is None:
        config = OrchestratorConfig()
    
    orchestrator = MultiModelOrchestrator(config)
    await orchestrator.start()
    
    return orchestrator