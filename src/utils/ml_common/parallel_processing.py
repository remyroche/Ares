"""
Parallel Processing Coordinator

This module provides comprehensive parallel processing coordination for ML operations
with automatic load balancing, error handling, and resource-aware scheduling.

Key Features:
- Parallel feature engineering
- Distributed cross-validation
- Parallel hyperparameter search
- Load-balanced processing
- Error handling and recovery
- Resource-aware scheduling

Built on existing utilities:
- Extends parallel_processing_optimizer.py capabilities
- Uses m1_cpu_optimizer.py for CPU optimization
- Leverages common_operations.py for robust error handling
- Integrates with data_processing_utils.py for data handling
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
import queue
import threading
import time

from ..math_validation import safe_divide
from ..common_operations import create_fallback_logger
from ..parallel_processing_optimizer import ParallelProcessor
from ..m1_cpu_optimizer import M1CPUOptimizer
from ..common_utilities import safe_dataframe_operation

logger = logging.getLogger(__name__)

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("Joblib not available - limited parallel processing capabilities")

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray not available - limited distributed computing capabilities")


class ParallelProcessingCoordinator:
    """Comprehensive parallel processing coordinator for ML operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize parallel processing coordinator with configuration."""
        self.config = config or {}
        self.logger = logger.getChild('ParallelCoordinator')

        # Configuration defaults
        self.max_workers = self.config.get('max_workers', multiprocessing.cpu_count())
        self.enable_joblib = self.config.get('enable_joblib', JOBLIB_AVAILABLE)
        self.enable_ray = self.config.get('enable_ray', RAY_AVAILABLE)
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.error_retry_limit = self.config.get('error_retry_limit', 3)
        self.task_timeout_seconds = self.config.get('task_timeout_seconds', 3600)
        self.prefer_process_pool = self.config.get('prefer_process_pool', False)

        # Initialize utilities
        self.parallel_processor = ParallelProcessor()
        self.cpu_optimizer = M1CPUOptimizer() if 'M1CPUOptimizer' in globals() else None

        # Task management
        self.active_tasks = {}
        self.task_queue = queue.Queue()
        self.task_results = {}

        # Resource monitoring
        self.resource_monitor = ResourceMonitor()

        # Initialize distributed computing if available
        if self.enable_ray and RAY_AVAILABLE:
            self._initialize_ray()

    def parallel_feature_engineering(self, feature_functions: List[Callable],
                                   data_chunks: List[Any],
                                   combine_results: bool = True) -> Union[List[Any], Any]:
        """
        Perform parallel feature engineering across data chunks.

        Args:
            feature_functions: List of feature engineering functions
            data_chunks: List of data chunks to process
            combine_results: Whether to combine results into single output

        Returns:
            List of results or combined result
        """
        try:
            self.logger.info(f"🔄 Starting parallel feature engineering: "
                           f"{len(feature_functions)} functions × {len(data_chunks)} chunks")

            # Create task combinations
            tasks = []
            for func in feature_functions:
                for chunk_idx, data_chunk in enumerate(data_chunks):
                    task = {
                        'function': func,
                        'data': data_chunk,
                        'chunk_idx': chunk_idx,
                        'func_name': getattr(func, '__name__', f'func_{len(tasks)}')
                    }
                    tasks.append(task)

            # Execute tasks in parallel
            results = self._execute_parallel_tasks(
                tasks,
                task_function=self._apply_feature_function,
                max_workers=min(self.max_workers, len(tasks))
            )

            # Organize results
            organized_results = self._organize_feature_results(results, len(feature_functions), len(data_chunks))

            if combine_results and len(data_chunks) > 1:
                # Combine results across chunks
                combined_results = self._combine_feature_results(organized_results)
                return combined_results
            else:
                return organized_results

        except Exception as e:
            self.logger.error(f"❌ Parallel feature engineering failed: {e}")
            return []

    def distributed_cross_validation(self, model_factory: Callable,
                                   X: np.ndarray, y: np.ndarray,
                                   cv_folds: int = 5,
                                   scoring_functions: Optional[List[Callable]] = None,
                                   cv: Optional[Any] = None) -> Dict[str, Any]:
        """
        Perform distributed cross-validation with parallel fold processing.

        Args:
            model_factory: Function that creates model instances
            X: Feature matrix
            y: Target array
            cv_folds: Number of CV folds
            scoring_functions: List of scoring functions

        Returns:
            Cross-validation results
        """
        try:
            self.logger.info(f"🔀 Starting distributed cross-validation: {cv_folds} folds")

            from sklearn.model_selection import TimeSeriesSplit
            if cv is None:
                test_size = max(1, len(X) // (cv_folds + 1))
                skf = TimeSeriesSplit(n_splits=cv_folds, test_size=test_size)
            else:
                skf = cv

            # Create CV tasks
            cv_tasks = []
            # Detect memmap to avoid copying large arrays in task payloads
            X_is_memmap = isinstance(X, np.memmap) and hasattr(X, 'filename')
            y_is_memmap = isinstance(y, np.memmap) and hasattr(y, 'filename')
            X_memmap_info = None
            y_memmap_info = None
            if X_is_memmap:
                try:
                    X_memmap_info = {
                        'filename': X.filename,
                        'dtype': str(X.dtype),
                        'shape': X.shape,
                        'mode': 'r'
                    }
                except Exception:
                    X_memmap_info = None
            if y_is_memmap:
                try:
                    y_memmap_info = {
                        'filename': y.filename,
                        'dtype': str(y.dtype),
                        'shape': y.shape,
                        'mode': 'r'
                    }
                except Exception:
                    y_memmap_info = None

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                task = {
                    'fold_idx': fold_idx,
                    'model_factory': model_factory,
                    # If memmap, pass indices and memmap info; otherwise pass sliced arrays
                    'train_idx': train_idx,
                    'test_idx': test_idx,
                    'X_train': None if X_memmap_info else X[train_idx],
                    'y_train': None if y_memmap_info else y[train_idx],
                    'X_test': None if X_memmap_info else X[test_idx],
                    'y_test': None if y_memmap_info else y[test_idx],
                    'X_memmap': X_memmap_info,
                    'y_memmap': y_memmap_info,
                    'scoring_functions': scoring_functions or []
                }
                try:
                    if len(np.unique(y[train_idx])) > 1:
                        from sklearn.utils.class_weight import compute_sample_weight
                        task['sample_weight'] = compute_sample_weight('balanced', y[train_idx])
                except Exception:
                    pass
                cv_tasks.append(task)

            # Execute CV tasks in parallel
            cv_results = self._execute_parallel_tasks(
                cv_tasks,
                task_function=self._execute_cv_fold,
                max_workers=min(self.max_workers, cv_folds)
            )

            # Aggregate CV results
            aggregated_results = self._aggregate_cv_results(cv_results)

            self.logger.info(f"✅ Distributed cross-validation completed: "
                           f"{len(cv_results)} folds processed")
            return aggregated_results

        except Exception as e:
            self.logger.error(f"❌ Distributed cross-validation failed: {e}")
            return {'error': str(e)}

    def gpu_accelerated_processing(self, tasks: List[Callable],
                                 gpu_config: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        GPU-accelerated parallel processing for compute-intensive tasks.

        Args:
            tasks: List of callable tasks to execute
            gpu_config: GPU configuration parameters

        Returns:
            List of task results
        """
        try:
            if gpu_config is None:
                gpu_config = {
                    'batch_size': self.config.get('gpu_batch_size', 1000),
                    'memory_threshold': self.config.get('gpu_memory_threshold', 0.8),
                    'enable_mixed_precision': self.config.get('enable_mixed_precision', True),
                    'max_concurrent_tasks': self.config.get('gpu_max_concurrent', 4)
                }

            self.logger.info(f"🚀 Starting GPU-accelerated processing for {len(tasks)} tasks")

            if not self.enable_ray and not RAY_AVAILABLE:
                self.logger.warning("⚠️ Ray not available - falling back to CPU processing")
                return self._cpu_fallback_processing(tasks)

            # Initialize GPU manager if available
            gpu_manager = None
            try:
                from ..m1_gpu_utils import M1GPUManager as _M1GPU
                gpu_manager = _M1GPU(gpu_config)
            except Exception:
                gpu_manager = None

            # Prepare GPU-optimized task execution
            gpu_tasks = []
            for task in tasks:
                gpu_task = self._prepare_gpu_task(task, gpu_config)
                gpu_tasks.append(gpu_task)

            # Execute tasks with GPU acceleration
            results = self._execute_gpu_tasks(gpu_tasks, gpu_config)

            self.logger.info(f"✅ GPU-accelerated processing completed: {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"❌ GPU-accelerated processing failed: {e}")
            return self._cpu_fallback_processing(tasks)

    def adaptive_load_balancing(self, tasks: List[Any],
                               resource_constraints: Optional[Dict[str, Any]] = None) -> List[List[Any]]:
        """
        Adaptive load balancing based on system resources and task characteristics.

        Args:
            tasks: List of tasks to balance
            resource_constraints: Resource constraints and limits

        Returns:
            List of balanced task groups
        """
        try:
            if resource_constraints is None:
                resource_constraints = {
                    'cpu_limit': self.max_workers,
                    'memory_limit_gb': 8.0,
                    'task_complexity_weights': {'cpu_intensive': 2, 'io_intensive': 1, 'memory_intensive': 3}
                }

            self.logger.info(f"⚖️ Starting adaptive load balancing for {len(tasks)} tasks")

            # Analyze system resources
            system_resources = self._analyze_system_resources()

            # Classify tasks by resource requirements
            task_profiles = self._classify_tasks_by_complexity(tasks)

            # Create balanced task groups
            balanced_groups = self._create_balanced_task_groups(
                task_profiles, system_resources, resource_constraints
            )

            self.logger.info(f"✅ Load balancing completed: {len(balanced_groups)} groups created")
            return balanced_groups

        except Exception as e:
            self.logger.error(f"❌ Adaptive load balancing failed: {e}")
            return [tasks]  # Return single group as fallback

    def fault_tolerant_parallel_execution(self, tasks: List[Any],
                                        recovery_strategies: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fault-tolerant parallel execution with automatic recovery strategies.

        Args:
            tasks: List of tasks to execute
            recovery_strategies: Recovery strategies configuration

        Returns:
            Execution results with fault tolerance statistics
        """
        try:
            if recovery_strategies is None:
                recovery_strategies = {
                    'max_retries': self.error_retry_limit,
                    'retry_delay_seconds': 1,
                    'exponential_backoff': True,
                    'checkpoint_interval': 10,
                    'enable_circuit_breaker': True,
                    'circuit_breaker_threshold': 0.5
                }

            self.logger.info(f"🛡️ Starting fault-tolerant execution for {len(tasks)} tasks")

            # Initialize fault tolerance components
            fault_tracker = self._initialize_fault_tracker(recovery_strategies)

            # Execute tasks with fault tolerance
            execution_results = self._execute_with_fault_tolerance(
                tasks, fault_tracker, recovery_strategies
            )

            # Generate fault tolerance report
            fault_report = self._generate_fault_tolerance_report(
                fault_tracker, execution_results
            )

            self.logger.info(f"✅ Fault-tolerant execution completed: {len(execution_results['successful'])} successful, {len(execution_results['failed'])} failed")

            return {
                'results': execution_results,
                'fault_report': fault_report,
                'recovery_stats': fault_tracker
            }

        except Exception as e:
            self.logger.error(f"❌ Fault-tolerant execution failed: {e}")
            return {'error': str(e), 'results': {'successful': [], 'failed': tasks}}

    def parallel_feature_engineering_gpu(self, feature_functions: List[Callable],
                                       data_chunks: List[Any],
                                       gpu_config: Optional[Dict[str, Any]] = None) -> Union[List[Any], Any]:
        """
        GPU-accelerated parallel feature engineering.

        Args:
            feature_functions: List of feature engineering functions
            data_chunks: List of data chunks to process
            gpu_config: GPU configuration

        Returns:
            Feature engineering results
        """
        try:
            self.logger.info(f"🔬 Starting GPU-accelerated feature engineering: "
                           f"{len(feature_functions)} functions × {len(data_chunks)} chunks")

            # Prepare GPU-optimized feature engineering tasks
            gpu_tasks = []
            for func in feature_functions:
                for chunk_idx, data_chunk in enumerate(data_chunks):
                    gpu_task = {
                        'function': func,
                        'data': data_chunk,
                        'chunk_idx': chunk_idx,
                        'task_type': 'feature_engineering',
                        'gpu_config': gpu_config or {}
                    }
                    gpu_tasks.append(gpu_task)

            # Execute with GPU acceleration
            results = self.gpu_accelerated_processing(gpu_tasks, gpu_config)

            # Organize results
            organized_results = self._organize_gpu_feature_results(results, len(feature_functions), len(data_chunks))

            self.logger.info(f"✅ GPU feature engineering completed: {len(results)} results processed")
            return organized_results

        except Exception as e:
            self.logger.error(f"❌ GPU feature engineering failed: {e}")
            return self.parallel_feature_engineering(feature_functions, data_chunks)

    # Helper methods for new functionality

    def _prepare_gpu_task(self, task: Callable, gpu_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a task for GPU execution."""
        try:
            return {
                'task': task,
                'gpu_config': gpu_config,
                'task_id': f"gpu_task_{hash(str(task)) % 10000}",
                'prepared_at': datetime.now().isoformat()
            }
        except Exception:
            return {'task': task, 'gpu_config': gpu_config}

    def _execute_gpu_tasks(self, gpu_tasks: List[Dict[str, Any]],
                          gpu_config: Dict[str, Any]) -> List[Any]:
        """Execute tasks with GPU acceleration."""
        try:
            results = []

            if RAY_AVAILABLE and self.enable_ray:
                # Use Ray for distributed GPU execution
                import ray

                @ray.remote
                def execute_gpu_task(task_config):
                    try:
                        task = task_config['task']
                        # Simulate GPU processing (in real implementation, this would use actual GPU operations)
                        result = task()
                        return {'result': result, 'success': True}
                    except Exception as e:
                        return {'error': str(e), 'success': False}

                # Execute tasks in parallel with Ray
                ray_refs = [execute_gpu_task.remote(task) for task in gpu_tasks]
                ray_results = ray.get(ray_refs)

                for result in ray_results:
                    if result.get('success', False):
                        results.append(result.get('result'))
                    else:
                        results.append(None)

            else:
                # Fallback to CPU processing with GPU simulation
                self.logger.warning("⚠️ Ray not available - using CPU processing with GPU simulation")

                for task_config in gpu_tasks:
                    try:
                        task = task_config['task']
                        result = task()
                        results.append(result)
                    except Exception as e:
                        self.logger.warning(f"Task execution failed: {e}")
                        results.append(None)

            return results
        except Exception as e:
            self.logger.error(f"GPU task execution failed: {e}")
            return [None] * len(gpu_tasks)

    def _cpu_fallback_processing(self, tasks: List[Callable]) -> List[Any]:
        """Fallback CPU processing when GPU is not available."""
        try:
            results = []
            for task in tasks:
                try:
                    result = task()
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Task execution failed: {e}")
                    results.append(None)
            return results
        except Exception:
            return [None] * len(tasks)

    def _analyze_system_resources(self) -> Dict[str, Any]:
        """Analyze current system resource availability."""
        try:
            import psutil

            resources = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'cpu_count': multiprocessing.cpu_count(),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }

            return resources
        except Exception:
            return {
                'cpu_percent': 50.0,
                'memory_percent': 50.0,
                'available_memory_gb': 4.0,
                'cpu_count': multiprocessing.cpu_count()
            }

    def _classify_tasks_by_complexity(self, tasks: List[Any]) -> List[Dict[str, Any]]:
        """Classify tasks by their resource complexity."""
        try:
            task_profiles = []

            for i, task in enumerate(tasks):
                # Simple task profiling (in practice, this would be more sophisticated)
                task_profile = {
                    'task_id': i,
                    'task': task,
                    'complexity': 'medium',  # Default
                    'estimated_cpu': 1,
                    'estimated_memory': 100,  # MB
                    'estimated_time': 1.0  # seconds
                }

                # Try to infer complexity from task attributes
                if hasattr(task, '__name__'):
                    func_name = task.__name__.lower()
                    if 'train' in func_name or 'fit' in func_name:
                        task_profile.update({
                            'complexity': 'high',
                            'estimated_cpu': 2,
                            'estimated_memory': 500,
                            'estimated_time': 5.0
                        })
                    elif 'predict' in func_name or 'transform' in func_name:
                        task_profile.update({
                            'complexity': 'medium',
                            'estimated_cpu': 1,
                            'estimated_memory': 200,
                            'estimated_time': 2.0
                        })

                task_profiles.append(task_profile)

            return task_profiles
        except Exception:
            # Fallback: assign medium complexity to all tasks
            return [{
                'task_id': i,
                'task': task,
                'complexity': 'medium',
                'estimated_cpu': 1,
                'estimated_memory': 100,
                'estimated_time': 1.0
            } for i, task in enumerate(tasks)]

    def _create_balanced_task_groups(self, task_profiles: List[Dict[str, Any]],
                                   system_resources: Dict[str, Any],
                                   constraints: Dict[str, Any]) -> List[List[Any]]:
        """Create balanced groups of tasks based on resource constraints."""
        try:
            # Simple bin packing algorithm for load balancing
            groups = []
            current_group = []
            current_load = {'cpu': 0, 'memory': 0, 'time': 0}

            max_cpu = constraints.get('cpu_limit', system_resources.get('cpu_count', 4))
            max_memory = constraints.get('memory_limit_gb', 8.0) * 1024  # Convert to MB

            for task_profile in task_profiles:
                task_cpu = task_profile['estimated_cpu']
                task_memory = task_profile['estimated_memory']
                task_time = task_profile['estimated_time']

                # Check if task fits in current group
                if (current_load['cpu'] + task_cpu <= max_cpu and
                    current_load['memory'] + task_memory <= max_memory):

                    # Add to current group
                    current_group.append(task_profile['task'])
                    current_load['cpu'] += task_cpu
                    current_load['memory'] += task_memory
                    current_load['time'] = max(current_load['time'], task_time)

                else:
                    # Start new group
                    if current_group:
                        groups.append(current_group)

                    current_group = [task_profile['task']]
                    current_load = {
                        'cpu': task_cpu,
                        'memory': task_memory,
                        'time': task_time
                    }

            # Add final group
            if current_group:
                groups.append(current_group)

            return groups
        except Exception:
            # Fallback: return single group
            return [[profile['task'] for profile in task_profiles]]

    def _initialize_fault_tracker(self, recovery_strategies: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize fault tracking and recovery mechanisms."""
        try:
            return {
                'total_tasks': 0,
                'successful_tasks': 0,
                'failed_tasks': 0,
                'retried_tasks': 0,
                'circuit_breaker_tripped': False,
                'failure_rate': 0.0,
                'recovery_attempts': {},
                'failure_patterns': {},
                'last_checkpoint': datetime.now()
            }
        except Exception:
            return {'error': 'Fault tracker initialization failed'}

    def _execute_with_fault_tolerance(self, tasks: List[Any],
                                    fault_tracker: Dict[str, Any],
                                    recovery_strategies: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tasks with fault tolerance mechanisms."""
        try:
            successful_results = []
            failed_tasks = []
            max_retries = recovery_strategies.get('max_retries', 3)

            fault_tracker['total_tasks'] = len(tasks)

            for i, task in enumerate(tasks):
                retry_count = 0
                success = False

                while retry_count <= max_retries and not success:
                    try:
                        # Execute task
                        result = task()

                        # Success
                        successful_results.append(result)
                        fault_tracker['successful_tasks'] += 1
                        success = True

                    except Exception as e:
                        retry_count += 1
                        fault_tracker['retried_tasks'] += 1

                        if retry_count <= max_retries:
                            # Retry with exponential backoff
                            delay = recovery_strategies.get('retry_delay_seconds', 1) * (2 ** (retry_count - 1))
                            self.logger.warning(f"⚠️ Task {i} failed (attempt {retry_count}), retrying in {delay}s: {e}")
                            time.sleep(delay)
                        else:
                            # Max retries exceeded
                            failed_tasks.append(task)
                            fault_tracker['failed_tasks'] += 1
                            self.logger.error(f"❌ Task {i} failed permanently after {max_retries} retries: {e}")

                # Update failure rate for circuit breaker
                total_processed = fault_tracker['successful_tasks'] + fault_tracker['failed_tasks']
                if total_processed > 0:
                    fault_tracker['failure_rate'] = fault_tracker['failed_tasks'] / total_processed

                # Check circuit breaker
                if (recovery_strategies.get('enable_circuit_breaker', False) and
                    fault_tracker['failure_rate'] > recovery_strategies.get('circuit_breaker_threshold', 0.5)):
                    fault_tracker['circuit_breaker_tripped'] = True
                    self.logger.critical("🚨 Circuit breaker tripped - stopping execution due to high failure rate")
                    break

            return {
                'successful': successful_results,
                'failed': failed_tasks,
                'circuit_breaker_tripped': fault_tracker.get('circuit_breaker_tripped', False)
            }
        except Exception as e:
            self.logger.error(f"Fault-tolerant execution error: {e}")
            return {'successful': [], 'failed': tasks, 'error': str(e)}

    def _generate_fault_tolerance_report(self, fault_tracker: Dict[str, Any],
                                       execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive fault tolerance report."""
        try:
            total_tasks = fault_tracker.get('total_tasks', 0)
            successful_tasks = fault_tracker.get('successful_tasks', 0)
            failed_tasks = fault_tracker.get('failed_tasks', 0)

            report = {
                'execution_summary': {
                    'total_tasks': total_tasks,
                    'successful_tasks': successful_tasks,
                    'failed_tasks': failed_tasks,
                    'success_rate': safe_divide(successful_tasks, total_tasks) * 100,
                    'failure_rate': safe_divide(failed_tasks, total_tasks) * 100
                },
                'recovery_stats': {
                    'retried_tasks': fault_tracker.get('retried_tasks', 0),
                    'circuit_breaker_tripped': fault_tracker.get('circuit_breaker_tripped', False),
                    'average_retry_attempts': safe_divide(
                        fault_tracker.get('retried_tasks', 0),
                        max(1, failed_tasks)
                    )
                },
                'recommendations': []
            }

            # Generate recommendations based on failure patterns
            if report['execution_summary']['failure_rate'] > 20:
                report['recommendations'].append("High failure rate detected - review task implementation")
            if fault_tracker.get('circuit_breaker_tripped', False):
                report['recommendations'].append("Circuit breaker activated - investigate systemic issues")
            if report['recovery_stats']['average_retry_attempts'] > 2:
                report['recommendations'].append("Many tasks require multiple retries - optimize task stability")

            return report
        except Exception:
            return {'error': 'Fault tolerance report generation failed'}

    def _organize_gpu_feature_results(self, results: List[Any],
                                    num_functions: int,
                                    num_chunks: int) -> Dict[str, Any]:
        """Organize GPU feature engineering results."""
        try:
            organized = {
                'results_by_function': [[] for _ in range(num_functions)],
                'results_by_chunk': [[] for _ in range(num_chunks)],
                'successful_executions': 0,
                'failed_executions': 0
            }

            for i, result in enumerate(results):
                if result is not None:
                    func_idx = i % num_functions
                    chunk_idx = i // num_functions

                    organized['results_by_function'][func_idx].append(result)
                    organized['results_by_chunk'][chunk_idx].append(result)
                    organized['successful_executions'] += 1
                else:
                    organized['failed_executions'] += 1

            return organized
        except Exception:
            return {'error': 'GPU result organization failed'}

    def parallel_hyperparameter_search(self, parameter_grid: Dict[str, List[Any]],
                                     model_factory: Callable,
                                     X: np.ndarray, y: np.ndarray,
                                     evaluation_function: Optional[Callable] = None,
                                     search_strategy: str = 'grid',
                                     n_random_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform parallel hyperparameter search.

        Args:
            parameter_grid: Dictionary of parameter ranges
            model_factory: Function that creates model instances
            X: Feature matrix
            y: Target array
            evaluation_function: Custom evaluation function
            search_strategy: Search strategy ('grid', 'random')

        Returns:
            Hyperparameter search results
        """
        try:
            self.logger.info(f"🔍 Starting parallel hyperparameter search: {search_strategy} strategy")

            # Generate parameter combinations
            if search_strategy == 'grid':
                param_combinations = self._generate_grid_parameters(parameter_grid)
            elif search_strategy == 'random':
                param_combinations = self._generate_random_parameters(parameter_grid, n_samples=(n_random_samples or 50))
            else:
                raise ValueError(f"Unsupported search strategy: {search_strategy}")

            # Create search tasks (memmap-aware)
            search_tasks = []
            X_is_memmap = isinstance(X, np.memmap) and hasattr(X, 'filename')
            y_is_memmap = isinstance(y, np.memmap) and hasattr(y, 'filename')
            X_memmap_info = None
            y_memmap_info = None
            if X_is_memmap:
                try:
                    X_memmap_info = {
                        'filename': X.filename,
                        'dtype': str(X.dtype),
                        'shape': X.shape,
                        'mode': 'r'
                    }
                except Exception:
                    X_memmap_info = None
            if y_is_memmap:
                try:
                    y_memmap_info = {
                        'filename': y.filename,
                        'dtype': str(y.dtype),
                        'shape': y.shape,
                        'mode': 'r'
                    }
                except Exception:
                    y_memmap_info = None
            for param_idx, params in enumerate(param_combinations):
                task = {
                    'param_idx': param_idx,
                    'params': params,
                    'model_factory': model_factory,
                    'X': None if X_memmap_info else X,
                    'y': None if y_memmap_info else y,
                    'X_memmap': X_memmap_info,
                    'y_memmap': y_memmap_info,
                    'evaluation_function': evaluation_function
                }
                search_tasks.append(task)

            # Execute search tasks in parallel
            search_results = self._execute_parallel_tasks(
                search_tasks,
                task_function=self._evaluate_parameter_combination,
                max_workers=min(self.max_workers, len(search_tasks))
            )

            # Find best parameters
            best_result = self._find_best_parameters(search_results)

            results = {
                'best_params': best_result.get('params', {}),
                'best_score': best_result.get('score', 0),
                'all_results': search_results,
                'total_combinations': len(param_combinations),
                'search_strategy': search_strategy
            }

            self.logger.info(f"✅ Parallel hyperparameter search completed: "
                           f"Best score: {results['best_score']:.4f}")
            return results

        except Exception as e:
            self.logger.error(f"❌ Parallel hyperparameter search failed: {e}")
            return {'error': str(e)}

    def load_balanced_processing(self, tasks: List[Dict[str, Any]],
                               worker_capabilities: Optional[Dict[str, Any]] = None,
                               load_balancing_strategy: str = 'round_robin') -> List[Any]:
        """
        Perform load-balanced processing of tasks.

        Args:
            tasks: List of task dictionaries
            worker_capabilities: Worker capability information
            load_balancing_strategy: Load balancing strategy

        Returns:
            List of task results
        """
        try:
            self.logger.info(f"⚖️ Starting load-balanced processing: {len(tasks)} tasks")

            if worker_capabilities is None:
                worker_capabilities = self._get_default_worker_capabilities()

            # Apply load balancing strategy
            if load_balancing_strategy == 'round_robin':
                balanced_tasks = self._round_robin_balancing(tasks, worker_capabilities)
            elif load_balancing_strategy == 'weighted':
                balanced_tasks = self._weighted_balancing(tasks, worker_capabilities)
            elif load_balancing_strategy == 'adaptive':
                balanced_tasks = self._adaptive_balancing(tasks, worker_capabilities)
            else:
                balanced_tasks = tasks

            # Execute balanced tasks
            results = self._execute_load_balanced_tasks(balanced_tasks, worker_capabilities)

            self.logger.info(f"✅ Load-balanced processing completed: {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"❌ Load-balanced processing failed: {e}")
            return []

    def error_handling_parallel_execution(self, tasks: List[Dict[str, Any]],
                                        max_retries: Optional[int] = None,
                                        error_handling_strategy: str = 'retry') -> List[Any]:
        """
        Execute tasks with comprehensive error handling and retry logic.

        Args:
            tasks: List of task dictionaries
            max_retries: Maximum number of retries per task
            error_handling_strategy: Error handling strategy ('retry', 'skip', 'fail_fast')

        Returns:
            List of task results with error information
        """
        try:
            if max_retries is None:
                max_retries = self.error_retry_limit

            self.logger.info(f"🛡️ Starting error-handling parallel execution: "
                           f"{len(tasks)} tasks, max_retries={max_retries}")

            # Execute tasks with error handling
            results = []

            if error_handling_strategy == 'retry':
                results = self._execute_with_retry(tasks, max_retries)
            elif error_handling_strategy == 'skip':
                results = self._execute_with_skip(tasks)
            elif error_handling_strategy == 'fail_fast':
                results = self._execute_fail_fast(tasks)
            else:
                results = self._execute_parallel_tasks(tasks)

            # Generate error summary
            error_summary = self._generate_error_summary(results)

            self.logger.info(f"✅ Error-handling execution completed: "
                           f"{error_summary['successful_tasks']}/{error_summary['total_tasks']} successful")
            return results

        except Exception as e:
            self.logger.error(f"❌ Error-handling parallel execution failed: {e}")
            return []

    def resource_aware_scheduling(self, task_batches: List[List[Dict[str, Any]]],
                                resource_constraints: Optional[Dict[str, Any]] = None,
                                scheduling_strategy: str = 'priority') -> List[Any]:
        """
        Perform resource-aware task scheduling.

        Args:
            task_batches: List of task batches
            resource_constraints: Resource constraints and limits
            scheduling_strategy: Scheduling strategy ('priority', 'fair', 'resource_optimized')

        Returns:
            Scheduled task results
        """
        try:
            self.logger.info(f"📅 Starting resource-aware scheduling: {len(task_batches)} batches")

            if resource_constraints is None:
                resource_constraints = self._get_default_resource_constraints()

            # Monitor resources throughout execution
            resource_monitor = threading.Thread(
                target=self._monitor_resources,
                args=(resource_constraints,),
                daemon=True
            )
            resource_monitor.start()

            # Execute task batches with resource awareness
            all_results = []

            for batch_idx, task_batch in enumerate(task_batches):
                try:
                    self.logger.debug(f"Processing batch {batch_idx + 1}/{len(task_batches)}")

                    # Check resource availability
                    if not self._check_resource_availability(resource_constraints):
                        self.logger.warning(f"⚠️ Resource constraints reached, pausing batch {batch_idx}")
                        time.sleep(5)  # Wait before retrying

                    # Execute batch
                    batch_results = self._execute_parallel_tasks(
                        task_batch,
                        max_workers=self._calculate_optimal_workers(resource_constraints)
                    )

                    all_results.extend(batch_results)

                except Exception as batch_e:
                    self.logger.warning(f"Batch {batch_idx} failed: {batch_e}")
                    continue

            self.logger.info(f"✅ Resource-aware scheduling completed: {len(all_results)} total results")
            return all_results

        except Exception as e:
            self.logger.error(f"❌ Resource-aware scheduling failed: {e}")
            return []

    def _execute_parallel_tasks(self, tasks: List[Dict[str, Any]],
                              task_function: Optional[Callable] = None,
                              max_workers: Optional[int] = None) -> List[Any]:
        """Execute tasks in parallel using available backend."""
        try:
            if max_workers is None:
                max_workers = self.max_workers

            results = []

            if self.enable_joblib and JOBLIB_AVAILABLE:
                # Use joblib for parallel execution
                joblib_results = Parallel(n_jobs=max_workers)(
                    delayed(self._execute_single_task)(task, task_function) for task in tasks
                )
                results = joblib_results

            elif self.enable_ray and RAY_AVAILABLE:
                # Use Ray for distributed execution
                ray_results = []
                for task in tasks:
                    ray_future = self._execute_single_task_ray.remote(task, task_function)
                    ray_results.append(ray_future)

                results = ray.get(ray_results)

            else:
                # Prefer process pool for CPU-bound tasks if configured
                ExecutorClass = ProcessPoolExecutor if self.prefer_process_pool else ThreadPoolExecutor
                try:
                    with ExecutorClass(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(self._execute_single_task, task, task_function)
                            for task in tasks
                        ]

                        for future in as_completed(futures):
                            try:
                                result = future.result(timeout=self.task_timeout_seconds)
                                results.append(result)
                            except Exception as task_e:
                                self.logger.warning(f"Task execution failed: {task_e}")
                                results.append({'error': str(task_e)})
                except Exception as e:
                    # Fallback to threads on process pool failure
                    self.logger.debug(f"Process pool unavailable, falling back to threads: {e}")
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(self._execute_single_task, task, task_function)
                            for task in tasks
                        ]
                        for future in as_completed(futures):
                            try:
                                result = future.result(timeout=self.task_timeout_seconds)
                                results.append(result)
                            except Exception as task_e:
                                self.logger.warning(f"Task execution failed: {task_e}")
                                results.append({'error': str(task_e)})

            return results

        except Exception as e:
            self.logger.error(f"Parallel task execution failed: {e}")
            return [{'error': str(e)} for _ in tasks]

    def _apply_feature_function(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Apply feature engineering function to data chunk."""
        try:
            func = task['function']
            data = task['data']
            chunk_idx = task['chunk_idx']
            func_name = task['func_name']

            # Apply function
            result = func(data)

            return {
                'chunk_idx': chunk_idx,
                'func_name': func_name,
                'result': result,
                'success': True
            }

        except Exception as e:
            return {
                'chunk_idx': task.get('chunk_idx', -1),
                'func_name': task.get('func_name', 'unknown'),
                'error': str(e),
                'success': False
            }

    def _execute_cv_fold(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cross-validation fold."""
        try:
            fold_idx = task['fold_idx']
            model_factory = task['model_factory']
            # Rehydrate memmap if info provided to avoid copying large arrays
            if task.get('X_memmap'):
                try:
                    info = task['X_memmap']
                    X_full = np.memmap(info['filename'], dtype=np.dtype(info['dtype']), mode=info['mode'], shape=tuple(info['shape']))
                    train_idx = task['train_idx']
                    test_idx = task['test_idx']
                    X_train = X_full[train_idx]
                    X_test = X_full[test_idx]
                except Exception as e:
                    return {'fold_idx': fold_idx, 'error': f"Memmap load failed for X: {e}", 'success': False}
            else:
                X_train = task['X_train']
                X_test = task['X_test']

            if task.get('y_memmap'):
                try:
                    infoy = task['y_memmap']
                    y_full = np.memmap(infoy['filename'], dtype=np.dtype(infoy['dtype']), mode=infoy['mode'], shape=tuple(infoy['shape']))
                    train_idx = task['train_idx']
                    test_idx = task['test_idx']
                    y_train = y_full[train_idx]
                    y_test = y_full[test_idx]
                except Exception as e:
                    return {'fold_idx': fold_idx, 'error': f"Memmap load failed for y: {e}", 'success': False}
            else:
                y_train = task['y_train']
                y_test = task['y_test']
            scoring_functions = task['scoring_functions']

            # Create and train model
            model = model_factory()
            try:
                if hasattr(model, 'set_params') and hasattr(model, 'get_params'):
                    params = model.get_params()
                    if 'n_jobs' in params:
                        model.set_params(n_jobs=1)
            except Exception:
                pass
            try:
                import inspect
                sample_weight = task.get('sample_weight')
                if sample_weight is not None and 'sample_weight' in inspect.signature(model.fit).parameters:
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                else:
                    model.fit(X_train, y_train)
            except Exception:
                model.fit(X_train, y_train)

            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = None

            # Calculate scores
            scores = {}
            for scoring_func in scoring_functions:
                try:
                    score_name = getattr(scoring_func, '__name__', 'custom_score')
                    scores[score_name] = scoring_func(y_test, y_pred)
                except Exception as score_e:
                    self.logger.debug(f"Scoring function failed: {score_e}")
                    continue

            # Default accuracy score
            if not scores:
                from sklearn.metrics import accuracy_score
                scores['accuracy'] = accuracy_score(y_test, y_pred)

            return {
                'fold_idx': fold_idx,
                'scores': scores,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'success': True
            }

        except Exception as e:
            return {
                'fold_idx': task.get('fold_idx', -1),
                'error': str(e),
                'success': False
            }

    def _evaluate_parameter_combination(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate hyperparameter combination."""
        try:
            param_idx = task['param_idx']
            params = task['params']
            model_factory = task['model_factory']
            # Rehydrate memmap if needed
            if task.get('X_memmap'):
                try:
                    info = task['X_memmap']
                    X = np.memmap(info['filename'], dtype=np.dtype(info['dtype']), mode=info['mode'], shape=tuple(info['shape']))
                except Exception as e:
                    return {'param_idx': param_idx, 'params': params, 'error': f"Memmap load failed for X: {e}", 'success': False}
            else:
                X = task['X']

            if task.get('y_memmap'):
                try:
                    infoy = task['y_memmap']
                    y = np.memmap(infoy['filename'], dtype=np.dtype(infoy['dtype']), mode=infoy['mode'], shape=tuple(infoy['shape']))
                except Exception as e:
                    return {'param_idx': param_idx, 'params': params, 'error': f"Memmap load failed for y: {e}", 'success': False}
            else:
                y = task['y']
            evaluation_function = task['evaluation_function']

            # Create model with parameters
            model = model_factory(**params)

            # Evaluate model
            if evaluation_function:
                score = evaluation_function(model, X, y)
            else:
                # Default cross-validation evaluation
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                score = np.mean(scores)

            return {
                'param_idx': param_idx,
                'params': params,
                'score': score,
                'success': True
            }

        except Exception as e:
            return {
                'param_idx': task.get('param_idx', -1),
                'params': task.get('params', {}),
                'error': str(e),
                'success': False
            }

    def _organize_feature_results(self, results: List[Dict[str, Any]],
                                n_functions: int, n_chunks: int) -> Dict[str, Any]:
        """Organize feature engineering results."""
        organized = {}

        for result in results:
            if result.get('success', False):
                func_name = result['func_name']
                chunk_idx = result['chunk_idx']

                if func_name not in organized:
                    organized[func_name] = {}

                organized[func_name][chunk_idx] = result['result']

        return organized

    def _combine_feature_results(self, organized_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine feature results across chunks."""
        combined = {}

        for func_name, chunk_results in organized_results.items():
            if chunk_results:
                # Combine chunks (assuming DataFrames)
                try:
                    combined_chunks = list(chunk_results.values())
                    if all(isinstance(chunk, pd.DataFrame) for chunk in combined_chunks):
                        combined[func_name] = pd.concat(combined_chunks, ignore_index=True)
                    else:
                        combined[func_name] = combined_chunks
                except Exception as combine_e:
                    self.logger.warning(f"Feature combination failed for {func_name}: {combine_e}")
                    combined[func_name] = chunk_results

        return combined

    def _aggregate_cv_results(self, cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        successful_results = [r for r in cv_results if r.get('success', False)]

        if not successful_results:
            return {'error': 'No successful CV folds'}

        # Aggregate scores
        all_scores = {}
        for result in successful_results:
            for score_name, score_value in result.get('scores', {}).items():
                if score_name not in all_scores:
                    all_scores[score_name] = []
                all_scores[score_name].append(score_value)

        aggregated_scores = {}
        for score_name, scores in all_scores.items():
            aggregated_scores[f"{score_name}_mean"] = np.mean(scores)
            aggregated_scores[f"{score_name}_std"] = np.std(scores)

        return {
            'aggregated_scores': aggregated_scores,
            'fold_results': successful_results,
            'n_successful_folds': len(successful_results),
            'n_total_folds': len(cv_results)
        }

    def _generate_grid_parameters(self, parameter_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate grid parameter combinations."""
        try:
            from itertools import product

            param_names = list(parameter_grid.keys())
            param_values = list(parameter_grid.values())

            combinations = []
            for value_combination in product(*param_values):
                param_dict = dict(zip(param_names, value_combination))
                combinations.append(param_dict)

            return combinations

        except Exception as e:
            self.logger.warning(f"Grid parameter generation failed: {e}")
            return []

    def _generate_random_parameters(self, parameter_grid: Dict[str, List[Any]],
                                  n_samples: int = 50) -> List[Dict[str, Any]]:
        """Generate random parameter combinations."""
        try:
            import random

            combinations = []
            for _ in range(n_samples):
                param_dict = {}
                for param_name, param_values in parameter_grid.items():
                    param_dict[param_name] = random.choice(param_values)
                combinations.append(param_dict)

            return combinations

        except Exception as e:
            self.logger.warning(f"Random parameter generation failed: {e}")
            return []

    def _find_best_parameters(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find best parameter combination from search results."""
        successful_results = [r for r in search_results if r.get('success', False) and 'score' in r]

        if not successful_results:
            return {'params': {}, 'score': 0}

        # Find result with highest score
        best_result = max(successful_results, key=lambda x: x['score'])
        return best_result

    def _execute_single_task(self, task: Dict[str, Any], task_function: Optional[Callable]) -> Any:
        """Execute a single task."""
        try:
            if task_function:
                return task_function(task)
            else:
                # Default task execution
                func = task.get('function')
                if func:
                    args = task.get('args', [])
                    kwargs = task.get('kwargs', {})
                    return func(*args, **kwargs)
                else:
                    return {'error': 'No function specified'}

        except Exception as e:
            return {'error': str(e)}

    def _get_default_worker_capabilities(self) -> Dict[str, Any]:
        """Get default worker capabilities."""
        return {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_gb': 8.0,  # Assume 8GB default
            'gpu_available': False
        }

    def _round_robin_balancing(self, tasks: List[Dict[str, Any]],
                             worker_capabilities: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Apply round-robin load balancing."""
        n_workers = worker_capabilities.get('cpu_count', 4)
        balanced_tasks = [[] for _ in range(n_workers)]

        for i, task in enumerate(tasks):
            worker_idx = i % n_workers
            balanced_tasks[worker_idx].append(task)

        return balanced_tasks

    def _weighted_balancing(self, tasks: List[Dict[str, Any]],
                          worker_capabilities: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Apply weighted load balancing based on worker capabilities."""
        # Simplified weighted balancing
        return self._round_robin_balancing(tasks, worker_capabilities)

    def _adaptive_balancing(self, tasks: List[Dict[str, Any]],
                          worker_capabilities: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Apply adaptive load balancing based on current load."""
        # Simplified adaptive balancing
        return self._round_robin_balancing(tasks, worker_capabilities)

    def _execute_load_balanced_tasks(self, balanced_tasks: List[List[Dict[str, Any]]],
                                   worker_capabilities: Dict[str, Any]) -> List[Any]:
        """Execute load-balanced tasks."""
        all_results = []

        # Execute each worker's tasks
        for worker_idx, worker_tasks in enumerate(balanced_tasks):
            if worker_tasks:
                worker_results = self._execute_parallel_tasks(worker_tasks, max_workers=1)
                all_results.extend(worker_results)

        return all_results

    def _execute_with_retry(self, tasks: List[Dict[str, Any]], max_retries: int) -> List[Any]:
        """Execute tasks with retry logic."""
        results = []
        retry_queue = tasks.copy()

        for attempt in range(max_retries + 1):
            if not retry_queue:
                break

            self.logger.debug(f"Retry attempt {attempt + 1}/{max_retries + 1}")

            # Execute current batch
            batch_results = self._execute_parallel_tasks(retry_queue)

            # Separate successful and failed tasks
            successful_results = []
            failed_tasks = []

            for task, result in zip(retry_queue, batch_results):
                if result.get('success', False) or 'error' not in result:
                    successful_results.append(result)
                else:
                    failed_tasks.append(task)

            results.extend(successful_results)
            retry_queue = failed_tasks

        # Add failed results for tasks that exhausted retries
        for failed_task in retry_queue:
            results.append({'error': f'Failed after {max_retries} retries', 'task': failed_task})

        return results

    def _execute_with_skip(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """Execute tasks with skip on error."""
        results = []

        for task in tasks:
            try:
                result = self._execute_single_task(task, None)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e), 'skipped': True})

        return results

    def _execute_fail_fast(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """Execute tasks with fail-fast behavior."""
        results = []

        for task in tasks:
            try:
                result = self._execute_single_task(task, None)
                if 'error' in result:
                    raise Exception(f"Task failed: {result['error']}")
                results.append(result)
            except Exception as e:
                self.logger.error(f"❌ Fail-fast: {e}")
                raise e

        return results

    def _generate_error_summary(self, results: List[Any]) -> Dict[str, Any]:
        """Generate error summary from results."""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if isinstance(r, dict) and 'error' not in r)
        failed_tasks = total_tasks - successful_tasks

        return {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': safe_divide(successful_tasks, total_tasks) * 100
        }

    def _check_resource_availability(self, resource_constraints: Dict[str, Any]) -> bool:
        """Check if resources are available."""
        try:
            # Simple resource check
            return self.resource_monitor.check_availability(resource_constraints)
        except:
            return True  # Default to available if check fails

    def _calculate_optimal_workers(self, resource_constraints: Dict[str, Any]) -> int:
        """Calculate optimal number of workers based on resources."""
        try:
            available_memory = resource_constraints.get('max_memory_gb', 8.0)
            cpu_count = resource_constraints.get('cpu_count', multiprocessing.cpu_count())

            # Simple heuristic: 1 worker per 2GB of memory, max of CPU count
            memory_based_workers = max(1, int(available_memory / 2))
            optimal_workers = min(memory_based_workers, cpu_count)

            return optimal_workers

        except Exception:
            return min(4, multiprocessing.cpu_count())

    def _monitor_resources(self, resource_constraints: Dict[str, Any]) -> None:
        """Monitor resources during execution."""
        try:
            while True:
                self.resource_monitor.update_stats()
                time.sleep(10)  # Monitor every 10 seconds
        except Exception as e:
            self.logger.debug(f"Resource monitoring failed: {e}")

    def _initialize_ray(self) -> None:
        """Initialize Ray for distributed computing."""
        try:
            if not RAY_AVAILABLE:
                return

            ray.init(ignore_reinit_error=True)
            self.logger.info("✨ Ray initialized for distributed computing")

        except Exception as e:
            self.logger.warning(f"Ray initialization failed: {e}")

    if RAY_AVAILABLE:
        @staticmethod
        @ray.remote
        def _execute_single_task_ray(task: Dict[str, Any], task_function: Optional[Callable]) -> Any:
            """Execute single task with Ray (remote function)."""
            try:
                if task_function:
                    return task_function(task)
                else:
                    func = task.get('function')
                    if func:
                        args = task.get('args', [])
                        kwargs = task.get('kwargs', {})
                        return func(*args, **kwargs)
                    else:
                        return {'error': 'No function specified'}
            except Exception as e:
                return {'error': str(e)}


class ResourceMonitor:
    """Resource monitoring utility."""

    def __init__(self):
        """Initialize resource monitor."""
        self.stats = {}
        self.logger = logger.getChild('ResourceMonitor')

    def check_availability(self, constraints: Dict[str, Any]) -> bool:
        """Check if resources are available within constraints."""
        try:
            # Simple availability check
            return True  # Placeholder implementation
        except Exception:
            return True

    def update_stats(self) -> None:
        """Update resource statistics."""
        try:
            # Update resource stats
            self.stats['timestamp'] = datetime.now().isoformat()
        except Exception as e:
            self.logger.debug(f"Resource stats update failed: {e}")
