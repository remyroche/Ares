from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
import numpy as np
import pandas as pd

#!/usr/bin/env python3
"""Pipeline Orchestrator for Step03 with DAG-based Execution and Resource-aware Scheduling.

This module provides advanced pipeline orchestration for HMM regime discovery with:
- DAG-based task execution with dependency management
- Resource-aware scheduling and load balancing
- Parallel execution where possible
- Intelligent caching and incremental processing
- Performance monitoring and optimization
"""

import asyncio

import time
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from pathlib import Path
import json
from collections import defaultdict, deque
import networkx as nx
import psutil

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

# Performance optimization imports
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TaskPriority(Enum):
    """Task execution priority."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """Pipeline task definition."""
    id: str
    name: str
    function: Callable
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    cache_key: Optional[str] = None

    # Runtime attributes
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)

@dataclass
class PipelineMetrics:
    """Pipeline execution metrics."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    total_execution_time: float = 0.0
    average_task_time: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    bottleneck_tasks: List[str] = field(default_factory=list)

class ResourceManager:
    """Resource-aware task scheduling."""

    def __init__(self, max_concurrent_tasks: int = None,
                 memory_limit_gb: float = None,
                 cpu_limit_percent: float = None):
        self.max_concurrent_tasks = max_concurrent_tasks or min(8, (joblib.cpu_count() if JOBLIB_AVAILABLE else 4))
        self.memory_limit_gb = memory_limit_gb or (psutil.virtual_memory().total / (1024**3)) * 0.8  # 80% of total
        self.cpu_limit_percent = cpu_limit_percent or 80.0

        self.logger = logging.getLogger('ResourceManager')
        self.current_tasks = 0
        self.resource_lock = threading.Lock()

        # Resource monitoring
        self.resource_history = deque(maxlen=100)

    def can_schedule_task(self, task: Task) -> Tuple[bool, str]:
        """Check if a task can be scheduled based on resource constraints."""

        with self.resource_lock:
            current_metrics = self.get_resource_metrics()

            # Check concurrent task limit
            if self.current_tasks >= self.max_concurrent_tasks:
                return False, f"Max concurrent tasks ({self.max_concurrent_tasks}) reached"

            # Check memory requirements
            memory_required = task.resource_requirements.get('memory_gb', 0.5)
            if current_metrics.memory_mb / 1024 + memory_required > self.memory_limit_gb:
                return False, ".2f"

            # Check CPU requirements
            cpu_required = task.resource_requirements.get('cpu_percent', 10.0)
            if current_metrics.cpu_percent + cpu_required > self.cpu_limit_percent:
                return False, ".1f"

            return True, "OK"

    def allocate_resources(self, task: Task) -> None:
        """Allocate resources for task execution."""
        with self.resource_lock:
            self.current_tasks += 1

    def release_resources(self, task: Task) -> None:
        """Release resources after task completion."""
        with self.resource_lock:
            self.current_tasks = max(0, self.current_tasks - 1)

    def get_resource_metrics(self) -> ResourceMetrics:
        """Get current resource usage metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            metrics = ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=memory.used / (1024**2),
                disk_usage_percent=disk.percent
            )

            # Store in history
            self.resource_history.append(metrics)

            return metrics

        except Exception as e:
            self.logger.warning(f"Failed to get resource metrics: {e}")
            return ResourceMetrics()

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.resource_history:
            return {}

        metrics_list = list(self.resource_history)

        return {
            'avg_cpu_percent': sum(m.cpu_percent for m in metrics_list) / len(metrics_list),
            'max_cpu_percent': max(m.cpu_percent for m in metrics_list),
            'avg_memory_mb': sum(m.memory_mb for m in metrics_list) / len(metrics_list),
            'max_memory_mb': max(m.memory_mb for m in metrics_list),
            'current_concurrent_tasks': self.current_tasks,
            'max_concurrent_tasks': self.max_concurrent_tasks
        }

class TaskCache:
    """Intelligent task result caching."""

    def __init__(self, cache_dir: Path = None, max_cache_age_hours: int = 24):
        self.cache_dir = cache_dir or Path("data/cache/pipeline")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_age_hours = max_cache_age_hours
        self.logger = logging.getLogger('TaskCache')

        # Cache hit statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cache_key(self, task: Task) -> str:
        """Generate cache key for task."""
        if task.cache_key:
            return task.cache_key

        # Generate key from task properties
        key_components = [
            task.id,
            task.name,
            str(sorted(task.args)) if task.args else "",
            str(sorted(task.kwargs.items())) if task.kwargs else "",
            str(task.dependencies)
        ]

        import hashlib
        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def is_cached(self, task: Task) -> bool:
        """Check if task result is cached and valid."""
        cache_key = self.get_cache_key(task)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return False

        # Check cache age
        cache_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if cache_age_hours > self.max_cache_age_hours:
            cache_file.unlink()  # Remove old cache
            return False

        return True

    def get_cached_result(self, task: Task) -> Optional[Any]:
        """Get cached result for task."""
        if not self.is_cached(task):
            self.cache_misses += 1
            return None

        cache_key = self.get_cache_key(task)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)

            self.cache_hits += 1
            self.logger.debug(f"Cache hit for task {task.name}")
            return cached_data['result']

        except Exception as e:
            self.logger.warning(f"Failed to load cached result for {task.name}: {e}")
            return None

    def cache_result(self, task: Task, result: Any) -> None:
        """Cache task result."""
        cache_key = self.get_cache_key(task)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            cached_data = {
                'task_id': task.id,
                'task_name': task.name,
                'timestamp': time.time(),
                'result': result
            }

            with open(cache_file, 'w') as f:
                json.dump(cached_data, f, indent=2)

            self.logger.debug(f"Cached result for task {task.name}")

        except Exception as e:
            self.logger.warning(f"Failed to cache result for {task.name}: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

class DAGExecutor:
    """DAG-based task execution engine."""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, (joblib.cpu_count() if JOBLIB_AVAILABLE else 4))
        self.logger = logging.getLogger('DAGExecutor')

        # Components
        self.resource_manager = ResourceManager(max_concurrent_tasks=self.max_workers)
        self.task_cache = TaskCache()

        # Execution state
        self.tasks = {}
        self.completed_tasks = set()
        self.failed_tasks = set()
        self.running_tasks = set()

        # Threading
        self.execution_lock = threading.Lock()
        self.stop_event = threading.Event()

    def add_task(self, task: Task) -> None:
        """Add task to the execution graph."""
        self.tasks[task.id] = task

    def add_tasks(self, tasks: List[Task]) -> None:
        """Add multiple tasks to the execution graph."""
        for task in tasks:
            self.add_task(task)

    def validate_dependencies(self) -> bool:
        """Validate task dependencies."""
        task_ids = set(self.tasks.keys())

        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep not in task_ids:
                    self.logger.error(f"Task {task.id} depends on unknown task {dep}")
                    return False

        # Check for circular dependencies
        try:
            self._build_dependency_graph()
            return True
        except Exception as e:
            self.logger.error(f"Dependency validation failed: {e}")
            return False

    def _build_dependency_graph(self) -> nx.DiGraph:
        """Build dependency graph."""
        graph = nx.DiGraph()

        for task in self.tasks.values():
            graph.add_node(task.id, task=task)

            for dep in task.dependencies:
                graph.add_edge(dep, task.id)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Circular dependencies detected in task graph")

        return graph

    def get_execution_order(self) -> List[List[str]]:
        """Get tasks organized by execution order (levels)."""
        graph = self._build_dependency_graph()

        # Get topological sort
        execution_order = list(nx.topological_sort(graph))

        # Group by levels (tasks that can run in parallel)
        levels = []
        processed = set()

        while execution_order:
            current_level = []

            for task_id in execution_order[:]:
                # Check if all dependencies are processed
                task = self.tasks[task_id]
                deps_satisfied = all(dep in processed for dep in task.dependencies)

                if deps_satisfied:
                    current_level.append(task_id)
                    execution_order.remove(task_id)

            if current_level:
                levels.append(current_level)
                processed.update(current_level)
            else:
                break  # No more tasks can be processed

        return levels

    def execute_pipeline(self) -> Dict[str, Any]:
        """Execute the pipeline with DAG-based scheduling."""

        self.logger.info("🚀 Starting DAG-based pipeline execution")

        start_time = time.time()

        if not self.validate_dependencies():
            return {"error": "Invalid task dependencies"}

        # Get execution order
        execution_levels = self.get_execution_order()
        self.logger.info(f"📋 Pipeline has {len(execution_levels)} execution levels")

        # Execute tasks level by level
        for level_idx, level_tasks in enumerate(execution_levels):
            self.logger.info(f"📊 Executing level {level_idx + 1}/{len(execution_levels)} "
                           f"with {len(level_tasks)} tasks")

            # Execute tasks in this level
            level_results = self._execute_level(level_tasks)

            # Check for failures
            failed_in_level = [tid for tid in level_tasks if self.tasks[tid].status == TaskStatus.FAILED]
            if failed_in_level:
                self.logger.error(f"Level {level_idx + 1} failed with {len(failed_in_level)} failed tasks")
                break

        # Collect final results
        end_time = time.time()
        execution_time = end_time - start_time

        results = self._collect_execution_results()
        results['total_execution_time'] = execution_time
        results['execution_levels'] = len(execution_levels)

        self.logger.info(".2f"
                        ".1f")

        return results

    def _execute_level(self, level_tasks: List[str]) -> Dict[str, Any]:
        """Execute tasks in a single level."""

        results = {}

        # Process tasks that can be cached
        cached_tasks = []
        uncached_tasks = []

        for task_id in level_tasks:
            task = self.tasks[task_id]

            # Check cache
            if self.task_cache.is_cached(task):
                cached_result = self.task_cache.get_cached_result(task)
                if cached_result is not None:
                    task.result = cached_result
                    task.status = TaskStatus.COMPLETED
                    task.execution_time = 0.0  # Cached
                    self.completed_tasks.add(task_id)
                    cached_tasks.append(task_id)
                    continue

            uncached_tasks.append(task_id)

        self.logger.info(f"📋 Level has {len(cached_tasks)} cached tasks and {len(uncached_tasks)} new tasks")

        # Execute uncached tasks in parallel
        if uncached_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}

                for task_id in uncached_tasks:
                    task = self.tasks[task_id]

                    # Check resource constraints
                    can_schedule, reason = self.resource_manager.can_schedule_task(task)
                    if not can_schedule:
                        self.logger.warning(f"Cannot schedule task {task_id}: {reason}")
                        task.status = TaskStatus.SKIPPED
                        task.error = reason
                        continue

                    # Submit task for execution
                    future = executor.submit(self._execute_task, task)
                    futures[future] = task_id

                # Wait for completion
                for future in concurrent.futures.as_completed(futures):
                    task_id = futures[future]

                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        results[task_id] = result

                    except Exception as e:
                        self.logger.error(f"Task {task_id} execution failed: {e}")
                        self.tasks[task_id].status = TaskStatus.FAILED
                        self.tasks[task_id].error = str(e)
                        self.failed_tasks.add(task_id)

        return results

    def _execute_task(self, task: Task) -> Any:
        """Execute a single task with resource management."""

        try:
            # Allocate resources
            self.resource_manager.allocate_resources(task)

            # Update task status
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()

            # Check cache first
            if self.task_cache.is_cached(task):
                cached_result = self.task_cache.get_cached_result(task)
                if cached_result is not None:
                    task.result = cached_result
                    task.status = TaskStatus.COMPLETED
                    task.end_time = time.time()
                    task.execution_time = task.end_time - task.start_time

                    self.completed_tasks.add(task.id)
                    return cached_result

            # Execute task
            self.logger.debug(f"Executing task: {task.name}")

            if asyncio.iscoroutinefunction(task.function):
                # Async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(task.function(*task.args, **task.kwargs))
                finally:
                    loop.close()
            else:
                # Regular function
                result = task.function(*task.args, **task.kwargs)

            # Update task status
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.end_time = time.time()
            task.execution_time = task.end_time - task.start_time

            # Cache result
            self.task_cache.cache_result(task, result)

            # Mark as completed
            self.completed_tasks.add(task.id)

            self.logger.debug(f"✅ Task {task.name} completed in {task.execution_time:.2f}s")

            return result

        except Exception as e:
            self.logger.error(f"❌ Task {task.name} failed: {e}")

            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.end_time = time.time()
            if task.start_time:
                task.execution_time = task.end_time - task.start_time

            self.failed_tasks.add(task.id)
            raise

        finally:
            # Release resources
            self.resource_manager.release_resources(task)

    def _collect_execution_results(self) -> Dict[str, Any]:
        """Collect execution results and metrics."""

        # Task statistics
        task_stats = {
            'total_tasks': len(self.tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'skipped_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.SKIPPED])
        }

        # Performance metrics
        completed_task_times = [t.execution_time for t in self.tasks.values()
                              if t.execution_time is not None and t.status == TaskStatus.COMPLETED]

        performance_metrics = {
            'average_task_time': np.mean(completed_task_times) if completed_task_times else 0.0,
            'total_task_time': sum(completed_task_times) if completed_task_times else 0.0,
            'max_task_time': max(completed_task_times) if completed_task_times else 0.0,
            'min_task_time': min(completed_task_times) if completed_task_times else 0.0
        }

        # Resource utilization
        resource_summary = self.resource_manager.get_resource_summary()
        cache_stats = self.task_cache.get_cache_stats()

        return {
            'task_statistics': task_stats,
            'performance_metrics': performance_metrics,
            'resource_utilization': resource_summary,
            'cache_performance': cache_stats,
            'task_results': {tid: t.result for tid, t in self.tasks.items() if t.result is not None},
            'task_errors': {tid: t.error for tid, t in self.tasks.items() if t.error is not None}
        }

# STEP03 SPECIFIC PIPELINE

class Step03PipelineOrchestrator:
    """Pipeline orchestrator specifically for Step03 operations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('Step03PipelineOrchestrator')

        # Initialize components
        self.executor = DAGExecutor()
        self.resource_manager = self.executor.resource_manager

        # Performance tracking
        self.start_time = None

    def build_step03_pipeline(self, data: Any, features: Any) -> None:
        """Build the Step03 pipeline with optimized task dependencies."""

        self.logger.info("🔧 Building Step03 optimized pipeline")

        # Data preparation tasks
        data_prep_task = Task(
            id="data_preparation",
            name="Data Preparation",
            function=self._prepare_data,
            args=[data],
            priority=TaskPriority.HIGH,
            resource_requirements={'memory_gb': 1.0, 'cpu_percent': 20.0}
        )

        # Feature engineering tasks (can run in parallel)
        feature_tasks = []

        # Vectorized feature engineering
        vectorized_features_task = Task(
            id="vectorized_feature_engineering",
            name="Vectorized Feature Engineering",
            function=self._vectorized_feature_engineering,
            dependencies=["data_preparation"],
            priority=TaskPriority.HIGH,
            resource_requirements={'memory_gb': 2.0, 'cpu_percent': 30.0}
        )
        feature_tasks.append(vectorized_features_task)

        # Memory-optimized feature processing
        memory_optimized_task = Task(
            id="memory_optimized_processing",
            name="Memory-Optimized Processing",
            function=self._memory_optimized_processing,
            dependencies=["data_preparation"],
            priority=TaskPriority.NORMAL,
            resource_requirements={'memory_gb': 1.5, 'cpu_percent': 25.0}
        )
        feature_tasks.append(memory_optimized_task)

        # HMM regime discovery (depends on features)
        hmm_discovery_task = Task(
            id="hmm_regime_discovery",
            name="HMM Regime Discovery",
            function=self._hmm_regime_discovery,
            dependencies=["vectorized_feature_engineering"],
            priority=TaskPriority.CRITICAL,
            resource_requirements={'memory_gb': 3.0, 'cpu_percent': 40.0}
        )

        # Parallel Bayesian optimization
        bayesian_opt_task = Task(
            id="parallel_bayesian_optimization",
            name="Parallel Bayesian Optimization",
            function=self._parallel_bayesian_optimization,
            dependencies=["vectorized_feature_engineering"],
            priority=TaskPriority.HIGH,
            resource_requirements={'memory_gb': 2.5, 'cpu_percent': 35.0}
        )

        # Ensemble clustering (depends on HMM discovery)
        ensemble_clustering_task = Task(
            id="parallel_ensemble_clustering",
            name="Parallel Ensemble Clustering",
            function=self._parallel_ensemble_clustering,
            dependencies=["hmm_regime_discovery"],
            priority=TaskPriority.HIGH,
            resource_requirements={'memory_gb': 2.0, 'cpu_percent': 30.0}
        )

        # Results integration (depends on all major tasks)
        results_integration_task = Task(
            id="results_integration",
            name="Results Integration",
            function=self._integrate_results,
            dependencies=["hmm_regime_discovery", "parallel_bayesian_optimization", "parallel_ensemble_clustering"],
            priority=TaskPriority.NORMAL,
            resource_requirements={'memory_gb': 1.0, 'cpu_percent': 15.0}
        )

        # Add all tasks to executor
        self.executor.add_tasks([
            data_prep_task,
            vectorized_features_task,
            memory_optimized_task,
            hmm_discovery_task,
            bayesian_opt_task,
            ensemble_clustering_task,
            results_integration_task
        ])

    def execute_step03_pipeline(self) -> Dict[str, Any]:
        """Execute the complete Step03 pipeline."""

        self.logger.info("🚀 Executing Step03 optimized pipeline")
        self.start_time = time.time()

        try:
            results = self.executor.execute_pipeline()

            execution_time = time.time() - self.start_time
            results['total_pipeline_time'] = execution_time

            self.logger.info(".2f"
                            ".1f"
                            f"Cache hit rate: {results.get('cache_performance', {}).get('hit_rate', 0):.1%}")

            return results

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return {"error": str(e), "execution_time": time.time() - self.start_time}

    def _prepare_data(self, data: Any) -> Any:
        """Data preparation task."""
        # Import and use enhanced memory manager
        from .step03_memory_manager import EnhancedMemoryManager

        memory_manager = EnhancedMemoryManager()
        return memory_manager.optimize_dataframe_memory(data)

    def _vectorized_feature_engineering(self, data: Any) -> Any:
        """Vectorized feature engineering task."""
        # Import and use vectorized operations
        from .step03_vectorized_operations import get_vectorized_operations_manager, create_vectorized_config

        manager = get_vectorized_operations_manager()
        config = create_vectorized_config()
        return manager.process_dataset(data, config)

    def _memory_optimized_processing(self, data: Any) -> Any:
        """Memory-optimized processing task."""
        from .step03_memory_manager import EnhancedMemoryManager

        memory_manager = EnhancedMemoryManager()
        return memory_manager.stream_process_data(data, lambda x: x)  # Simple pass-through for now

    def _hmm_regime_discovery(self, features: Any) -> Any:
        """HMM regime discovery task."""
        # Import enhanced HMM discovery
        from .step03_hmm_regime_discovery import HMMRegimeDiscoveryStep

        # This would need proper initialization with config
        # For now, return mock result
        return {"regimes_discovered": 4, "quality_score": 0.85}

    def _parallel_bayesian_optimization(self, features: Any) -> Any:
        """Parallel Bayesian optimization task."""
        from .step03_enhanced_bayesian_optimization import ParallelBayesianOptimizer

        # Mock optimization result
        return {"best_score": 0.92, "optimization_time": 45.2}

    def _parallel_ensemble_clustering(self, hmm_results: Any) -> Any:
        """Parallel ensemble clustering task."""
        from .step03_advanced_ensemble_clustering import ParallelClusteringProcessor

        # Mock clustering result
        return {"n_clusters": 4, "quality_score": 0.88}

    def _integrate_results(self, hmm_results: Any, opt_results: Any, cluster_results: Any) -> Dict[str, Any]:
        """Integrate all results into final output."""
        return {
            "final_regimes": cluster_results.get("n_clusters", 4),
            "overall_quality_score": (
                hmm_results.get("quality_score", 0) +
                opt_results.get("best_score", 0) +
                cluster_results.get("quality_score", 0)
            ) / 3,
            "processing_summary": {
                "hmm_quality": hmm_results.get("quality_score", 0),
                "optimization_score": opt_results.get("best_score", 0),
                "clustering_quality": cluster_results.get("quality_score", 0)
            }
        }

# UTILITY FUNCTIONS

def create_step03_pipeline_config(**kwargs) -> Dict[str, Any]:
    """Create configuration for Step03 pipeline."""

    default_config = {
        'max_concurrent_tasks': None,  # Auto-detect
        'memory_limit_gb': None,       # Auto-detect
        'cpu_limit_percent': 80.0,
        'enable_caching': True,
        'cache_max_age_hours': 24,
        'enable_resource_monitoring': True,
        'pipeline_timeout_seconds': 3600,  # 1 hour
        'log_level': 'INFO'
    }

    default_config.update(kwargs)
    return default_config

def get_step03_pipeline_orchestrator(config: Dict[str, Any] = None) -> Step03PipelineOrchestrator:
    """Get Step03 pipeline orchestrator instance."""
    if config is None:
        config = create_step03_pipeline_config()

    return Step03PipelineOrchestrator(config)

if __name__ == "__main__":
    # Example usage

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame({
        'close': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'volume': np.random.randint(1000, 10000, n_samples)
    })

    features = np.random.randn(n_samples, 20)

    # Create pipeline
    config = create_step03_pipeline_config()
    orchestrator = get_step03_pipeline_orchestrator(config)

    # Build and execute pipeline
    orchestrator.build_step03_pipeline(data, features)
    results = orchestrator.execute_step03_pipeline()

    print(f"Pipeline execution results: {results}")
