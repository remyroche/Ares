"""
ML Pipeline Orchestration Utilities

This module provides comprehensive ML pipeline orchestration utilities for coordinating
complex machine learning workflows with error handling, monitoring, and optimization.

Key Features:
- ML pipeline creation and management
- Pipeline execution monitoring
- Automated pipeline optimization
- Pipeline failure recovery
- Resource-aware pipeline scheduling
- Pipeline result aggregation

Built on existing utilities:
- Uses common_operations.py for robust error handling
- Leverages parallel_processing_optimizer.py for parallel execution
- Integrates with data_processing_utils.py for data handling
- Builds on existing pipeline patterns
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading

from ..common_operations import create_fallback_logger
from src.utils.ml_common.utils import ParallelProcessor
from src.utils.common_utilities import safe_dataframe_operation

logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepStatus(Enum):
    """Pipeline step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class PipelineStep:
    """Represents a single step in the ML pipeline."""
    name: str
    function: Callable
    args: Optional[List[Any]] = None
    kwargs: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[int] = None
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class PipelineExecution:
    """Represents a pipeline execution."""
    pipeline_id: str
    steps: Dict[str, PipelineStep]
    status: PipelineStatus = PipelineStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_execution_time: Optional[float] = None
    results: Dict[str, Any] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.results is None:
            self.results = {}
        if self.errors is None:
            self.errors = []

class MLPipelineOrchestrator:
    """Comprehensive ML pipeline orchestration system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ML pipeline orchestrator with configuration."""
        self.logger = logger.getChild('PipelineOrchestrator')
        self.logger.info("🚀 Initializing MLPipelineOrchestrator...")
        start_time = time.time()

        self.config = config or {}
        self.logger.info(f"📊 Configuration loaded with {len(self.config)} parameters")

        # Configuration defaults
        self.max_workers = self.config.get('max_workers', 4)
        self.enable_parallel = self.config.get('enable_parallel', True)
        self.default_timeout = self.config.get('default_timeout', 3600)
        self.retry_failed_steps = self.config.get('retry_failed_steps', True)
        self.enable_monitoring = self.config.get('enable_monitoring', True)

        self.logger.info(f"📊 Max workers: {self.max_workers}")
        self.logger.info(f"📊 Parallel execution: {self.enable_parallel}")
        self.logger.info(f"📊 Default timeout: {self.default_timeout}s")
        self.logger.info(f"📊 Retry failed steps: {self.retry_failed_steps}")
        self.logger.info(f"📊 Monitoring enabled: {self.enable_monitoring}")

        # Pipeline storage
        self.active_pipelines: Dict[str, PipelineExecution] = {}
        self.completed_pipelines: Dict[str, PipelineExecution] = {}
        self.logger.debug("✅ Pipeline storage initialized")

        # Monitoring
        self.monitoring_queue = queue.Queue()
        self.monitoring_thread = None
        self.logger.debug("✅ Monitoring system initialized")

        init_time = time.time() - start_time
        self.logger.info(f"✅ MLPipelineOrchestrator initialized in {init_time:.3f}s")

        # Initialize utilities
        self.parallel_processor = ParallelProcessor() if self.enable_parallel else None

        # Start monitoring if enabled
        if self.enable_monitoring:
            self._start_monitoring()

    def create_training_pipeline(self, steps_config: List[Dict[str, Any]],
                               error_handling: str = 'robust',
                               pipeline_id: Optional[str] = None) -> str:
        """
        Create a new ML training pipeline.

        Args:
            steps_config: List of step configurations
            error_handling: Error handling strategy ('robust', 'strict', 'permissive')
            pipeline_id: Optional pipeline ID

        Returns:
            Pipeline ID
        """
        try:
            if pipeline_id is None:
                pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(datetime.now()) % 1000}"

            self.logger.info(f"🔧 Creating ML pipeline: {pipeline_id}")

            # Create pipeline steps
            steps = {}
            for step_config in steps_config:
                step = PipelineStep(
                    name=step_config['name'],
                    function=step_config['function'],
                    args=step_config.get('args', []),
                    kwargs=step_config.get('kwargs', {}),
                    dependencies=step_config.get('dependencies', []),
                    max_retries=step_config.get('max_retries', 3),
                    timeout_seconds=step_config.get('timeout_seconds', self.default_timeout)
                )
                steps[step.name] = step

            # Create pipeline execution
            pipeline = PipelineExecution(
                pipeline_id=pipeline_id,
                steps=steps,
                status=PipelineStatus.PENDING
            )

            self.active_pipelines[pipeline_id] = pipeline

            self.logger.info(f"✅ Pipeline created: {pipeline_id} with {len(steps)} steps")
            return pipeline_id

        except Exception as e:
            self.logger.error(f"❌ Pipeline creation failed: {e}")
            raise

    async def execute_pipeline(self, pipeline_id: str,
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute a pipeline asynchronously.

        Args:
            pipeline_id: Pipeline ID to execute
            progress_callback: Optional progress callback function

        Returns:
            Pipeline execution results
        """
        try:
            if pipeline_id not in self.active_pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")

            pipeline = self.active_pipelines[pipeline_id]
            pipeline.status = PipelineStatus.RUNNING
            pipeline.started_at = datetime.now()

            self.logger.info(f"🚀 Starting pipeline execution: {pipeline_id}")

            # Execute pipeline steps
            execution_results = await self._execute_pipeline_steps(
                pipeline, progress_callback
            )

            # Update pipeline status
            pipeline.completed_at = datetime.now()
            pipeline.total_execution_time = (
                pipeline.completed_at - pipeline.started_at
            ).total_seconds()

            if execution_results['success']:
                pipeline.status = PipelineStatus.COMPLETED
                pipeline.results = execution_results['results']
                self.completed_pipelines[pipeline_id] = pipeline
                del self.active_pipelines[pipeline_id]
            else:
                pipeline.status = PipelineStatus.FAILED
                pipeline.errors = execution_results['errors']

            self.logger.info(f"✅ Pipeline execution completed: {pipeline_id} - "
                           f"Status: {pipeline.status.value}")
            return execution_results

        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {e}")
            if pipeline_id in self.active_pipelines:
                self.active_pipelines[pipeline_id].status = PipelineStatus.FAILED
                self.active_pipelines[pipeline_id].errors.append(str(e))
            raise

    def pipeline_execution_monitoring(self, pipeline_id: str,
                                    progress_callback: Optional[Callable] = None) -> Iterator[Dict[str, Any]]:
        """
        Monitor pipeline execution progress.

        Args:
            pipeline_id: Pipeline ID to monitor
            progress_callback: Optional progress callback

        Yields:
            Progress updates
        """
        try:
            while pipeline_id in self.active_pipelines:
                pipeline = self.active_pipelines[pipeline_id]

                progress = {
                    'pipeline_id': pipeline_id,
                    'status': pipeline.status.value,
                    'total_steps': len(pipeline.steps),
                    'completed_steps': sum(1 for step in pipeline.steps.values()
                                          if step.status == StepStatus.COMPLETED),
                    'running_steps': sum(1 for step in pipeline.steps.values()
                                        if step.status == StepStatus.RUNNING),
                    'failed_steps': sum(1 for step in pipeline.steps.values()
                                       if step.status == StepStatus.FAILED),
                    'pending_steps': sum(1 for step in pipeline.steps.values()
                                        if step.status == StepStatus.PENDING),
                    'execution_time': (datetime.now() - (pipeline.started_at or pipeline.created_at)).total_seconds()
                }

                if progress_callback:
                    progress_callback(progress)

                yield progress

                # Check if pipeline is completed
                if pipeline.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]:
                    break

                time.sleep(1)  # Update every second

        except Exception as e:
            self.logger.error(f"❌ Pipeline monitoring failed: {e}")
            yield {'error': str(e)}

    def automated_pipeline_optimization(self, pipeline: PipelineExecution,
                                     performance_target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically optimize pipeline performance.

        Args:
            pipeline: Pipeline execution to optimize
            performance_target: Performance targets

        Returns:
            Optimization recommendations
        """
        try:
            self.logger.info(f"⚡ Optimizing pipeline: {pipeline.pipeline_id}")

            optimization_results = {
                'optimization_opportunities': [],
                'performance_analysis': {},
                'recommendations': [],
                'estimated_improvements': {}
            }

            # Analyze step execution times
            step_times = {}
            for step_name, step in pipeline.steps.items():
                if step.execution_time:
                    step_times[step_name] = step.execution_time

            if step_times:
                # Find bottlenecks
                sorted_steps = sorted(step_times.items(), key=lambda x: x[1], reverse=True)
                total_time = sum(step_times.values())

                optimization_results['performance_analysis'] = {
                    'total_execution_time': total_time,
                    'slowest_step': sorted_steps[0][0],
                    'slowest_step_time': sorted_steps[0][1],
                    'fastest_step': sorted_steps[-1][0],
                    'fastest_step_time': sorted_steps[-1][1]
                }

                # Identify parallelization opportunities
                independent_steps = self._identify_independent_steps(pipeline.steps)
                if len(independent_steps) > 1:
                    optimization_results['optimization_opportunities'].append({
                        'type': 'parallelization',
                        'steps': independent_steps,
                        'estimated_speedup': min(len(independent_steps), self.max_workers)
                    })

                # Identify caching opportunities
                cacheable_steps = self._identify_cacheable_steps(pipeline.steps)
                if cacheable_steps:
                    optimization_results['optimization_opportunities'].append({
                        'type': 'caching',
                        'steps': cacheable_steps,
                        'estimated_speedup': 1.5
                    })

            # Generate recommendations
            optimization_results['recommendations'] = self._generate_optimization_recommendations(
                optimization_results['optimization_opportunities']
            )

            self.logger.info(f"✅ Pipeline optimization completed: "
                           f"{len(optimization_results['optimization_opportunities'])} opportunities found")
            return optimization_results

        except Exception as e:
            self.logger.error(f"❌ Pipeline optimization failed: {e}")
            return {'error': str(e)}

    def pipeline_failure_recovery(self, failed_pipeline: PipelineExecution,
                                recovery_strategy: str = 'retry') -> Dict[str, Any]:
        """
        Recover from pipeline failure.

        Args:
            failed_pipeline: Failed pipeline execution
            recovery_strategy: Recovery strategy ('retry', 'skip', 'recreate')

        Returns:
            Recovery results
        """
        try:
            self.logger.info(f"🔄 Recovering pipeline: {failed_pipeline.pipeline_id}")

            recovery_results = {
                'recovery_strategy': recovery_strategy,
                'recovered_steps': [],
                'failed_steps': [],
                'new_pipeline_id': None
            }

            if recovery_strategy == 'retry':
                # Retry failed steps
                recovery_results.update(self._retry_failed_steps(failed_pipeline))

            elif recovery_strategy == 'skip':
                # Skip failed steps and continue
                recovery_results.update(self._skip_failed_steps(failed_pipeline))

            elif recovery_strategy == 'recreate':
                # Recreate pipeline with fixes
                new_pipeline_id = self._recreate_pipeline(failed_pipeline)
                recovery_results['new_pipeline_id'] = new_pipeline_id

            self.logger.info(f"✅ Pipeline recovery completed using {recovery_strategy} strategy")
            return recovery_results

        except Exception as e:
            self.logger.error(f"❌ Pipeline recovery failed: {e}")
            return {'error': str(e)}

    def resource_aware_pipeline_scheduling(self, pipelines: List[PipelineExecution],
                                        resource_constraints: Optional[Dict[str, Any]] = None,
                                        scheduling_strategy: str = 'priority') -> List[PipelineExecution]:
        """
        Schedule pipelines with resource awareness.

        Args:
            pipelines: List of pipelines to schedule
            resource_constraints: Resource constraints
            scheduling_strategy: Scheduling strategy ('priority', 'fair', 'resource_optimized')

        Returns:
            Scheduled pipelines
        """
        try:
            self.logger.info(f"📅 Scheduling {len(pipelines)} pipelines with {scheduling_strategy} strategy")

            if resource_constraints is None:
                resource_constraints = {
                    'max_concurrent_pipelines': 2,
                    'cpu_limit': 80,
                    'memory_limit': 80
                }

            if scheduling_strategy == 'priority':
                # Sort by priority (assuming pipelines have priority attribute)
                scheduled = sorted(pipelines, key=lambda p: getattr(p, 'priority', 0), reverse=True)

            elif scheduling_strategy == 'fair':
                # Round-robin scheduling
                scheduled = pipelines.copy()

            elif scheduling_strategy == 'resource_optimized':
                # Optimize based on resource requirements
                scheduled = self._optimize_resource_scheduling(pipelines, resource_constraints)

            else:
                scheduled = pipelines.copy()

            self.logger.info(f"✅ Pipeline scheduling completed: {len(scheduled)} pipelines scheduled")
            return scheduled

        except Exception as e:
            self.logger.error(f"❌ Pipeline scheduling failed: {e}")
            return pipelines

    def pipeline_result_aggregation(self, multiple_pipeline_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple pipeline executions.

        Args:
            multiple_pipeline_results: List of pipeline results

        Returns:
            Aggregated results
        """
        try:
            self.logger.info(f"📊 Aggregating results from {len(multiple_pipeline_results)} pipelines")

            aggregated_results = {
                'total_pipelines': len(multiple_pipeline_results),
                'successful_pipelines': 0,
                'failed_pipelines': 0,
                'performance_summary': {},
                'error_summary': {},
                'best_pipeline': None,
                'consensus_results': {}
            }

            # Aggregate individual results
            all_metrics = []
            all_errors = []

            best_score = -float('inf')
            best_pipeline = None

            for result in multiple_pipeline_results:
                if result.get('success', False):
                    aggregated_results['successful_pipelines'] += 1

                    # Collect metrics
                    if 'metrics' in result:
                        all_metrics.append(result['metrics'])

                    # Find best performing pipeline
                    if 'final_score' in result:
                        if result['final_score'] > best_score:
                            best_score = result['final_score']
                            best_pipeline = result

                else:
                    aggregated_results['failed_pipelines'] += 1
                    if 'error' in result:
                        all_errors.append(result['error'])

            # Calculate performance summary
            if all_metrics:
                aggregated_results['performance_summary'] = self._calculate_performance_summary(all_metrics)
                aggregated_results['best_pipeline'] = best_pipeline

            # Error summary
            if all_errors:
                aggregated_results['error_summary'] = {
                    'total_errors': len(all_errors),
                    'unique_errors': list(set(all_errors)),
                    'most_common_error': max(set(all_errors), key=all_errors.count) if all_errors else None
                }

            self.logger.info(f"✅ Result aggregation completed: "
                           f"{aggregated_results['successful_pipelines']}/{aggregated_results['total_pipelines']} successful")
            return aggregated_results

        except Exception as e:
            self.logger.error(f"❌ Result aggregation failed: {e}")
            return {'error': str(e)}

    async def _execute_pipeline_steps(self, pipeline: PipelineExecution,
                                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute pipeline steps with dependency resolution."""
        try:
            results = {'success': True, 'results': {}, 'errors': []}

            # Create execution queue
            pending_steps = queue.Queue()
            completed_steps = set()
            running_steps = set()

            # Initialize with steps that have no dependencies
            for step_name, step in pipeline.steps.items():
                if not step.dependencies:
                    pending_steps.put(step_name)

            # Execute steps
            while not pending_steps.empty() or running_steps:
                # Get next step to execute
                if not pending_steps.empty():
                    step_name = pending_steps.get()
                    step = pipeline.steps[step_name]

                    # Check if all dependencies are satisfied
                    if self._dependencies_satisfied(step, completed_steps):
                        running_steps.add(step_name)
                        step_status = await self._execute_single_step_async(pipeline, step, results)
                        running_steps.remove(step_name)

                        if step_status == StepStatus.COMPLETED:
                            completed_steps.add(step_name)

                            # Add dependent steps to queue
                            for dep_step_name, dep_step in pipeline.steps.items():
                                if (
                                    dep_step_name not in completed_steps
                                    and dep_step_name not in running_steps
                                    and dep_step_name not in [pending_steps.queue[i] for i in range(pending_steps.qsize())]
                                ):
                                    if self._dependencies_satisfied(dep_step, completed_steps):
                                        pending_steps.put(dep_step_name)

                            if progress_callback:
                                await progress_callback({
                                    'pipeline_id': pipeline.pipeline_id,
                                    'completed_steps': len(completed_steps),
                                    'total_steps': len(pipeline.steps)
                                })
                        elif step_status == StepStatus.PENDING:
                            # Retry requested – place back into queue
                            pending_steps.put(step_name)
                        else:  # StepStatus.FAILED
                            results['success'] = False
                            completed_steps.add(step_name)
                            if progress_callback:
                                await progress_callback({
                                    'pipeline_id': pipeline.pipeline_id,
                                    'completed_steps': len(completed_steps),
                                    'total_steps': len(pipeline.steps)
                                })
                    else:
                        # Put back in queue if dependencies not satisfied
                        pending_steps.put(step_name)

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            # Check for failed steps
            failed_steps = [name for name, step in pipeline.steps.items()
                          if step.status == StepStatus.FAILED]

            if failed_steps:
                results['success'] = False
                results['errors'].extend([f"Step {name} failed" for name in failed_steps])

            return results

        except Exception as e:
            return {'success': False, 'errors': [str(e)]}

    async def _execute_single_step_async(self, pipeline: PipelineExecution,
                                       step: PipelineStep, results: Dict[str, Any]) -> StepStatus:
        """Execute a single pipeline step asynchronously."""
        try:
            step.status = StepStatus.RUNNING
            step.started_at = datetime.now()

            # Execute step with timeout
            if step.timeout_seconds:
                result = await asyncio.wait_for(
                    self._execute_step_function(step),
                    timeout=step.timeout_seconds
                )
            else:
                result = await self._execute_step_function(step)

            step.result = result
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now()
            step.execution_time = (step.completed_at - step.started_at).total_seconds()

            results['results'][step.name] = result
            return StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.completed_at = datetime.now()
            if step.started_at:
                step.execution_time = (step.completed_at - step.started_at).total_seconds()

            # Retry logic
            if self.retry_failed_steps and step.retry_count < step.max_retries:
                step.retry_count += 1
                step.status = StepStatus.PENDING
                self.logger.warning(f"⚠️ Step {step.name} failed, retrying ({step.retry_count}/{step.max_retries})")
                return StepStatus.PENDING
            else:
                results['errors'].append(f"Step {step.name} failed: {str(e)}")
                return StepStatus.FAILED

    async def _execute_step_function(self, step: PipelineStep) -> Any:
        """Execute step function."""
        if asyncio.iscoroutinefunction(step.function):
            return await step.function(*step.args, **step.kwargs)
        else:
            # Run in thread pool for synchronous functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, step.function, *step.args, **step.kwargs
            )

    def _dependencies_satisfied(self, step: PipelineStep, completed_steps: set) -> bool:
        """Check if step dependencies are satisfied."""
        if not step.dependencies:
            return True
        return all(dep in completed_steps for dep in step.dependencies)

    def _identify_independent_steps(self, steps: Dict[str, PipelineStep]) -> List[str]:
        """Identify steps that can be executed in parallel."""
        independent_steps = []

        for step_name, step in steps.items():
            if not step.dependencies:
                independent_steps.append(step_name)
            else:
                # Check if step depends only on completed steps or other independent steps
                dependent_on_independent = all(
                    dep_step in independent_steps or
                    not steps.get(dep_step, PipelineStep("", None)).dependencies
                    for dep_step in step.dependencies
                )
                if dependent_on_independent:
                    independent_steps.append(step_name)

        return independent_steps

    def _identify_cacheable_steps(self, steps: Dict[str, PipelineStep]) -> List[str]:
        """Identify steps that can benefit from caching."""
        cacheable_steps = []

        # Simple heuristic: steps with 'load', 'read', or 'fetch' in name
        for step_name in steps.keys():
            if any(keyword in step_name.lower() for keyword in ['load', 'read', 'fetch', 'download']):
                cacheable_steps.append(step_name)

        return cacheable_steps

    def _generate_optimization_recommendations(self, opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        for opportunity in opportunities:
            if opportunity['type'] == 'parallelization':
                recommendations.append(
                    f"Consider parallelizing {len(opportunity['steps'])} independent steps "
                    f"for up to {opportunity['estimated_speedup']}x speedup"
                )
            elif opportunity['type'] == 'caching':
                recommendations.append(
                    f"Add caching for {len(opportunity['steps'])} data loading steps "
                    f"for estimated {opportunity['estimated_speedup']}x speedup"
                )

        if not recommendations:
            recommendations.append("Pipeline is already well-optimized")

        return recommendations

    def _retry_failed_steps(self, pipeline: PipelineExecution) -> Dict[str, Any]:
        """Retry failed steps in pipeline."""
        failed_steps = [name for name, step in pipeline.steps.items()
                       if step.status == StepStatus.FAILED]

        recovery_results = {
            'recovered_steps': [],
            'still_failed_steps': []
        }

        for step_name in failed_steps:
            step = pipeline.steps[step_name]

            # Reset step status
            step.status = StepStatus.PENDING
            step.error = None
            step.retry_count = 0

            recovery_results['recovered_steps'].append(step_name)

        return recovery_results

    def _skip_failed_steps(self, pipeline: PipelineExecution) -> Dict[str, Any]:
        """Skip failed steps and mark as completed."""
        failed_steps = [name for name, step in pipeline.steps.items()
                       if step.status == StepStatus.FAILED]

        for step_name in failed_steps:
            pipeline.steps[step_name].status = StepStatus.SKIPPED

        return {
            'skipped_steps': failed_steps,
            'continuation_possible': True
        }

    def _recreate_pipeline(self, failed_pipeline: PipelineExecution) -> str:
        """Recreate pipeline with fixes."""
        # This would implement logic to recreate pipeline with fixes
        # For now, return a placeholder
        new_pipeline_id = f"recreated_{failed_pipeline.pipeline_id}"
        return new_pipeline_id

    def _optimize_resource_scheduling(self, pipelines: List[PipelineExecution],
                                    constraints: Dict[str, Any]) -> List[PipelineExecution]:
        """Optimize pipeline scheduling based on resources."""
        # Simple resource-aware scheduling
        # Sort by estimated resource requirements
        return sorted(pipelines, key=lambda p: len(p.steps))

    def _calculate_performance_summary(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance summary across pipelines."""
        if not all_metrics:
            return {}

        # Aggregate metrics
        summary = {}
        metric_names = set()

        for metrics in all_metrics:
            metric_names.update(metrics.keys())

        for metric_name in metric_names:
            values = [m.get(metric_name) for m in all_metrics if metric_name in m]
            if values:
                summary[metric_name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': (sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5
                }

        return summary

    def _start_monitoring(self) -> None:
        """Start monitoring thread."""
        if self.monitoring_thread is None:
            self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
            self.monitoring_thread.start()

    def _monitoring_worker(self) -> None:
        """Monitoring worker thread."""
        while True:
            try:
                # Process monitoring queue
                while not self.monitoring_queue.empty():
                    monitoring_data = self.monitoring_queue.get()
                    self._process_monitoring_data(monitoring_data)

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Monitoring worker error: {e}")
                time.sleep(10)

    def _process_monitoring_data(self, data: Dict[str, Any]) -> None:
        """Process monitoring data."""
        # This would implement monitoring data processing
        # For now, just log it
        self.logger.debug(f"Monitoring data: {data}")

    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline execution status."""
        pipeline = (self.active_pipelines.get(pipeline_id) or
                   self.completed_pipelines.get(pipeline_id))

        if pipeline:
            return {
                'pipeline_id': pipeline.pipeline_id,
                'status': pipeline.status.value,
                'created_at': pipeline.created_at.isoformat(),
                'started_at': pipeline.started_at.isoformat() if pipeline.started_at else None,
                'completed_at': pipeline.completed_at.isoformat() if pipeline.completed_at else None,
                'total_execution_time': pipeline.total_execution_time,
                'total_steps': len(pipeline.steps),
                'completed_steps': sum(1 for step in pipeline.steps.values()
                                      if step.status == StepStatus.COMPLETED),
                'failed_steps': sum(1 for step in pipeline.steps.values()
                                   if step.status == StepStatus.FAILED)
            }

        return None

    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel pipeline execution."""
        if pipeline_id in self.active_pipelines:
            self.active_pipelines[pipeline_id].status = PipelineStatus.CANCELLED
            self.logger.info(f"🛑 Pipeline cancelled: {pipeline_id}")
            return True

        return False
