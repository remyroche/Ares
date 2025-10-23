#!/usr/bin/env python3
"""
Enhanced ML Pipeline Integration Module

This module integrates all the enhanced monitoring, error detection, HPO monitoring,
testing, and reporting components into a unified ML pipeline system.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import json
import pickle
from pathlib import Path
from collections import defaultdict, deque
import warnings
import traceback

from src.utils.tprint import tprint
from src.utils.logger import get_logger

# Import enhanced components
from ..monitoring.enhanced_error_detector import (
    get_global_error_detector,
    detect_error,
    ErrorCategory,
    ErrorSeverity
)
from ..optimization.enhanced_hpo_monitor import (
    get_global_hpo_monitor,
    HPOStatus,
    TrialResult
)
from ..testing.enhanced_testing_framework import (
    get_global_testing_framework,
    TestType,
    TestStatus,
    TestConfiguration
)
from ..reporting.enhanced_reporting_system import (
    get_global_reporting_system,
    ReportType,
    AlertLevel
)

logger = get_logger("EnhancedMLPipelineIntegration")

class PipelineStage(Enum):
    """ML pipeline stages."""
    DATA_PREPARATION = "data_preparation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    HPO_OPTIMIZATION = "hpo_optimization"
    MODEL_VALIDATION = "model_validation"
    MODEL_DEPLOYMENT = "model_deployment"
    MONITORING = "monitoring"

class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"

@dataclass
class PipelineExecution:
    """Pipeline execution tracking."""
    execution_id: str
    pipeline_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: PipelineStatus = PipelineStatus.PENDING
    current_stage: Optional[PipelineStage] = None
    stage_results: Dict[PipelineStage, Dict[str, Any]] = field(default_factory=dict)
    error_count: int = 0
    warning_count: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedMLPipelineIntegration:
    """Enhanced ML pipeline integration system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced ML pipeline integration."""
        self.config = config or {}
        self.logger = logger.getChild('EnhancedMLPipelineIntegration')

        # Initialize enhanced components
        self.error_detector = get_global_error_detector(self.config.get('error_detection', {}))
        self.hpo_monitor = get_global_hpo_monitor(self.config.get('hpo_monitoring', {}))
        self.testing_framework = get_global_testing_framework(
            TestConfiguration(**self.config.get('testing', {}))
        )
        self.reporting_system = get_global_reporting_system(self.config.get('reporting', {}))

        # Pipeline tracking
        self.active_executions: Dict[str, PipelineExecution] = {}
        self.completed_executions: Dict[str, PipelineExecution] = {}
        self.pipeline_history: deque = deque(maxlen=1000)

        # Stage handlers
        self.stage_handlers: Dict[PipelineStage, Callable] = {}
        self.stage_validators: Dict[PipelineStage, Callable] = {}

        # Integration state
        self.integration_active = False
        self.integration_thread = None
        self.lock = threading.Lock()

        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.stage_timings: Dict[PipelineStage, List[float]] = defaultdict(list)

        self.logger.info("🔗 Enhanced ML Pipeline Integration initialized")

    def register_stage_handler(self, stage: PipelineStage, handler: Callable):
        """Register a handler for a pipeline stage."""
        try:
            self.stage_handlers[stage] = handler
            self.logger.info(f"📋 Registered handler for stage: {stage.value}")
        except Exception as e:
            error_context = {
                'component': 'pipeline_integration',
                'function': 'register_stage_handler',
                'stage': stage.value
            }
            detect_error(e, error_context)
            raise

    def register_stage_validator(self, stage: PipelineStage, validator: Callable):
        """Register a validator for a pipeline stage."""
        try:
            self.stage_validators[stage] = validator
            self.logger.info(f"✅ Registered validator for stage: {stage.value}")
        except Exception as e:
            error_context = {
                'component': 'pipeline_integration',
                'function': 'register_stage_validator',
                'stage': stage.value
            }
            detect_error(e, error_context)
            raise

    def execute_pipeline(self,
                        pipeline_name: str,
                        stages: List[PipelineStage],
                        execution_config: Optional[Dict[str, Any]] = None) -> str:
        """Execute a complete ML pipeline with enhanced monitoring."""
        try:
            execution_id = self._generate_execution_id(pipeline_name)

            # Create execution tracking
            execution = PipelineExecution(
                execution_id=execution_id,
                pipeline_name=pipeline_name,
                start_time=datetime.now(),
                status=PipelineStatus.RUNNING,
                metadata=execution_config or {}
            )

            with self.lock:
                self.active_executions[execution_id] = execution

            self.logger.info(f"🚀 Starting pipeline execution: {pipeline_name} ({execution_id})")

            # Generate initial report
            self.reporting_system.generate_report(
                ReportType.TRAINING_PROGRESS,
                f"Pipeline Execution Started: {pipeline_name}",
                {
                    'execution_id': execution_id,
                    'pipeline_name': pipeline_name,
                    'stages': [stage.value for stage in stages],
                    'start_time': execution.start_time.isoformat(),
                    'config': execution_config
                }
            )

            # Execute stages
            try:
                for stage in stages:
                    execution.current_stage = stage
                    stage_start_time = time.time()

                    self.logger.info(f"🔄 Executing stage: {stage.value}")

                    # Validate stage prerequisites
                    if stage in self.stage_validators:
                        validation_result = self._validate_stage_prerequisites(stage, execution)
                        if not validation_result['valid']:
                            raise ValueError(f"Stage validation failed: {validation_result['error']}")

                    # Execute stage
                    stage_result = self._execute_stage(stage, execution)
                    execution.stage_results[stage] = stage_result

                    # Track timing
                    stage_duration = time.time() - stage_start_time
                    self.stage_timings[stage].append(stage_duration)

                    # Check for errors
                    if stage_result.get('error_count', 0) > 0:
                        execution.error_count += stage_result['error_count']

                    if stage_result.get('warning_count', 0) > 0:
                        execution.warning_count += stage_result['warning_count']

                    # Generate stage report
                    self._generate_stage_report(stage, stage_result, execution)

                    # Check for pipeline failure conditions
                    if self._should_fail_pipeline(execution):
                        execution.status = PipelineStatus.FAILED
                        break

                # Complete pipeline
                if execution.status == PipelineStatus.RUNNING:
                    execution.status = PipelineStatus.COMPLETED
                    execution.end_time = datetime.now()

            except Exception as e:
                execution.status = PipelineStatus.FAILED
                execution.end_time = datetime.now()

                error_context = {
                    'component': 'pipeline_integration',
                    'function': 'execute_pipeline',
                    'execution_id': execution_id,
                    'pipeline_name': pipeline_name,
                    'current_stage': execution.current_stage.value if execution.current_stage else None
                }
                detect_error(e, error_context)

                # Generate failure report
                self.reporting_system.generate_report(
                    ReportType.ERROR_ANALYSIS,
                    f"Pipeline Execution Failed: {pipeline_name}",
                    {
                        'execution_id': execution_id,
                        'pipeline_name': pipeline_name,
                        'failed_stage': execution.current_stage.value if execution.current_stage else None,
                        'error_message': str(e),
                        'error_traceback': traceback.format_exc(),
                        'stage_results': execution.stage_results
                    }
                )

                raise

            finally:
                # Move to completed executions
                with self.lock:
                    self.completed_executions[execution_id] = execution
                    if execution_id in self.active_executions:
                        del self.active_executions[execution_id]
                    self.pipeline_history.append(execution)

                # Generate final report
                self._generate_final_pipeline_report(execution)

            self.logger.info(f"✅ Pipeline execution completed: {pipeline_name} - {execution.status.value}")
            return execution_id

        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {e}")
            raise

    def _execute_stage(self, stage: PipelineStage, execution: PipelineExecution) -> Dict[str, Any]:
        """Execute a single pipeline stage with enhanced monitoring."""
        try:
            stage_start_time = time.time()

            # Check if handler exists
            if stage not in self.stage_handlers:
                raise ValueError(f"No handler registered for stage: {stage.value}")

            handler = self.stage_handlers[stage]

            # Execute stage with error detection
            try:
                stage_result = handler(execution)

                # Validate stage result
                if not isinstance(stage_result, dict):
                    stage_result = {'result': stage_result}

                # Add execution metadata
                stage_result.update({
                    'stage': stage.value,
                    'execution_id': execution.execution_id,
                    'start_time': stage_start_time,
                    'end_time': time.time(),
                    'duration': time.time() - stage_start_time,
                    'error_count': 0,
                    'warning_count': 0
                })

                return stage_result

            except Exception as stage_error:
                # Detect and classify stage error
                error_context = {
                    'component': 'pipeline_integration',
                    'function': '_execute_stage',
                    'stage': stage.value,
                    'execution_id': execution.execution_id,
                    'pipeline_name': execution.pipeline_name
                }

                error_record = detect_error(stage_error, error_context)

                # Create stage result with error information
                stage_result = {
                    'stage': stage.value,
                    'execution_id': execution.execution_id,
                    'start_time': stage_start_time,
                    'end_time': time.time(),
                    'duration': time.time() - stage_start_time,
                    'error_count': 1,
                    'warning_count': 0,
                    'error': {
                        'error_type': error_record.context.error_type,
                        'error_message': error_record.context.error_message,
                        'error_category': error_record.category.value,
                        'error_severity': error_record.severity.value,
                        'suggested_actions': error_record.suggested_actions
                    }
                }

                # Create alert if critical
                if error_record.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
                    self.reporting_system.create_alert(
                        AlertLevel.ERROR if error_record.severity == ErrorSeverity.HIGH else AlertLevel.CRITICAL,
                        f"Pipeline Stage Failed: {stage.value}",
                        f"Stage {stage.value} failed in pipeline {execution.pipeline_name}: {error_record.context.error_message}",
                        "pipeline_execution",
                        {
                            'execution_id': execution.execution_id,
                            'stage': stage.value,
                            'error_category': error_record.category.value,
                            'suggested_actions': error_record.suggested_actions
                        }
                    )

                return stage_result

        except Exception as e:
            self.logger.error(f"❌ Stage execution failed: {e}")
            raise

    def _validate_stage_prerequisites(self, stage: PipelineStage, execution: PipelineExecution) -> Dict[str, Any]:
        """Validate stage prerequisites."""
        try:
            validator = self.stage_validators[stage]
            validation_result = validator(execution)

            if not isinstance(validation_result, dict):
                validation_result = {'valid': bool(validation_result)}

            return validation_result

        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation failed: {str(e)}"
            }

    def _should_fail_pipeline(self, execution: PipelineExecution) -> bool:
        """Determine if pipeline should fail based on error conditions."""
        try:
            # Check error thresholds
            max_errors = self.config.get('max_errors_per_pipeline', 10)
            max_critical_errors = self.config.get('max_critical_errors_per_pipeline', 3)

            if execution.error_count > max_errors:
                self.logger.error(f"❌ Pipeline failed: too many errors ({execution.error_count})")
                return True

            # Check for critical errors in recent stages
            recent_stages = list(execution.stage_results.keys())[-3:]  # Last 3 stages
            critical_errors = 0

            for stage in recent_stages:
                stage_result = execution.stage_results[stage]
                if 'error' in stage_result:
                    error_severity = stage_result['error'].get('error_severity', 'medium')
                    if error_severity == 'critical':
                        critical_errors += 1

            if critical_errors > max_critical_errors:
                self.logger.error(f"❌ Pipeline failed: too many critical errors ({critical_errors})")
                return True

            return False

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to check pipeline failure conditions: {e}")
            return False

    def _generate_stage_report(self, stage: PipelineStage, stage_result: Dict[str, Any], execution: PipelineExecution):
        """Generate report for a pipeline stage."""
        try:
            report_type = self._get_report_type_for_stage(stage)

            self.reporting_system.generate_report(
                report_type,
                f"Stage Completed: {stage.value}",
                {
                    'execution_id': execution.execution_id,
                    'pipeline_name': execution.pipeline_name,
                    'stage': stage.value,
                    'stage_result': stage_result,
                    'execution_metrics': execution.metrics,
                    'overall_error_count': execution.error_count,
                    'overall_warning_count': execution.warning_count
                }
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate stage report: {e}")

    def _generate_final_pipeline_report(self, execution: PipelineExecution):
        """Generate final pipeline execution report."""
        try:
            # Calculate final metrics
            total_duration = (execution.end_time - execution.start_time).total_seconds() if execution.end_time else 0

            final_metrics = {
                'total_duration': total_duration,
                'stages_completed': len(execution.stage_results),
                'error_count': execution.error_count,
                'warning_count': execution.warning_count,
                'success_rate': 1.0 if execution.status == PipelineStatus.COMPLETED else 0.0,
                'stage_timings': {
                    stage.value: self.stage_timings.get(stage, [])
                    for stage in execution.stage_results.keys()
                }
            }

            execution.metrics.update(final_metrics)

            # Generate comprehensive report
            self.reporting_system.generate_report(
                ReportType.COMPREHENSIVE,
                f"Pipeline Execution {execution.status.value.title()}: {execution.pipeline_name}",
                {
                    'execution_id': execution.execution_id,
                    'pipeline_name': execution.pipeline_name,
                    'status': execution.status.value,
                    'start_time': execution.start_time.isoformat(),
                    'end_time': execution.end_time.isoformat() if execution.end_time else None,
                    'total_duration': total_duration,
                    'stages_completed': len(execution.stage_results),
                    'stage_results': execution.stage_results,
                    'final_metrics': final_metrics,
                    'metadata': execution.metadata
                }
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate final pipeline report: {e}")

    def _get_report_type_for_stage(self, stage: PipelineStage) -> ReportType:
        """Get appropriate report type for a pipeline stage."""
        if stage == PipelineStage.HPO_OPTIMIZATION:
            return ReportType.HPO_OPTIMIZATION
        elif stage == PipelineStage.MODEL_VALIDATION:
            return ReportType.MODEL_VALIDATION
        elif stage == PipelineStage.MODEL_TRAINING:
            return ReportType.TRAINING_PROGRESS
        else:
            return ReportType.PERFORMANCE_METRICS

    def _generate_execution_id(self, pipeline_name: str) -> str:
        """Generate unique execution ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name_hash = hashlib.md5(pipeline_name.encode()).hexdigest()[:8]
        return f"{pipeline_name}_{timestamp}_{name_hash}"

    def get_pipeline_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a pipeline execution."""
        try:
            execution = self.active_executions.get(execution_id) or self.completed_executions.get(execution_id)
            if not execution:
                return None

            return {
                'execution_id': execution.execution_id,
                'pipeline_name': execution.pipeline_name,
                'status': execution.status.value,
                'current_stage': execution.current_stage.value if execution.current_stage else None,
                'start_time': execution.start_time.isoformat(),
                'end_time': execution.end_time.isoformat() if execution.end_time else None,
                'duration': (execution.end_time - execution.start_time).total_seconds() if execution.end_time else None,
                'stages_completed': len(execution.stage_results),
                'error_count': execution.error_count,
                'warning_count': execution.warning_count,
                'metrics': execution.metrics
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get pipeline status: {e}")
            return None

    def get_integration_summary(self) -> Dict[str, Any]:
        """Get comprehensive integration summary."""
        try:
            with self.lock:
                active_count = len(self.active_executions)
                completed_count = len(self.completed_executions)

                # Calculate success rate
                successful_executions = sum(
                    1 for e in self.completed_executions.values()
                    if e.status == PipelineStatus.COMPLETED
                )
                success_rate = successful_executions / max(1, completed_count)

                # Get component summaries
                error_summary = self.error_detector.get_error_summary()
                hpo_summary = self.hpo_monitor.get_monitoring_summary()
                test_summary = self.testing_framework.get_test_summary()
                report_summary = self.reporting_system.get_report_summary()

                return {
                    'integration_summary': {
                        'active_executions': active_count,
                        'completed_executions': completed_count,
                        'success_rate': success_rate,
                        'integration_active': self.integration_active
                    },
                    'component_summaries': {
                        'error_detection': error_summary,
                        'hpo_monitoring': hpo_summary,
                        'testing_framework': test_summary,
                        'reporting_system': report_summary
                    },
                    'performance_metrics': {
                        'stage_timings': {
                            stage.value: {
                                'count': len(timings),
                                'mean': np.mean(timings) if timings else 0,
                                'std': np.std(timings) if timings else 0,
                                'min': np.min(timings) if timings else 0,
                                'max': np.max(timings) if timings else 0
                            }
                            for stage, timings in self.stage_timings.items()
                        }
                    }
                }

        except Exception as e:
            self.logger.error(f"❌ Failed to get integration summary: {e}")
            return {'error': str(e)}

    def start_integration_monitoring(self):
        """Start integration monitoring."""
        if self.integration_active:
            return

        self.integration_active = True
        self.integration_thread = threading.Thread(target=self._integration_monitoring_loop, daemon=True)
        self.integration_thread.start()

        self.logger.info("🔗 Integration monitoring started")

    def stop_integration_monitoring(self):
        """Stop integration monitoring."""
        self.integration_active = False
        if self.integration_thread:
            self.integration_thread.join(timeout=5)

        self.logger.info("🔗 Integration monitoring stopped")

    def _integration_monitoring_loop(self):
        """Main integration monitoring loop."""
        while self.integration_active:
            try:
                # Monitor active executions
                self._monitor_active_executions()

                # Generate periodic reports
                self._generate_periodic_integration_reports()

                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"❌ Integration monitoring loop error: {e}")
                time.sleep(60)

    def _monitor_active_executions(self):
        """Monitor active pipeline executions."""
        try:
            current_time = datetime.now()
            timeout_threshold = timedelta(hours=24)  # 24 hour timeout

            for execution_id, execution in list(self.active_executions.items()):
                # Check for timeout
                if (current_time - execution.start_time) > timeout_threshold:
                    execution.status = PipelineStatus.FAILED
                    execution.end_time = current_time

                    # Move to completed
                    self.completed_executions[execution_id] = execution
                    del self.active_executions[execution_id]

                    # Create timeout alert
                    self.reporting_system.create_alert(
                        AlertLevel.CRITICAL,
                        "Pipeline Execution Timeout",
                        f"Pipeline {execution.pipeline_name} timed out after 24 hours",
                        "pipeline_monitor",
                        {'execution_id': execution_id}
                    )

                    self.logger.error(f"⏰ Pipeline timeout: {execution.pipeline_name}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to monitor active executions: {e}")

    def _generate_periodic_integration_reports(self):
        """Generate periodic integration reports."""
        try:
            # Generate system health report every hour
            current_time = datetime.now()
            if not hasattr(self, '_last_health_report') or \
               (current_time - self._last_health_report).total_seconds() > 3600:

                self.reporting_system.generate_report(
                    ReportType.SYSTEM_HEALTH,
                    "System Health Report",
                    self.get_integration_summary()
                )

                self._last_health_report = current_time

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate periodic reports: {e}")

# Global integration instance
_global_integration = None

def get_global_integration(config: Optional[Dict[str, Any]] = None) -> EnhancedMLPipelineIntegration:
    """Get or create global integration instance."""
    global _global_integration

    if _global_integration is None:
        _global_integration = EnhancedMLPipelineIntegration(config)
        _global_integration.start_integration_monitoring()

    return _global_integration
