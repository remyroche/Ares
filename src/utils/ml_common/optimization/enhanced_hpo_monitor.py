#!/usr/bin/env python3
"""
Enhanced HPO Monitoring and Failure Detection System

This module provides comprehensive monitoring, failure detection, and early stopping
mechanisms for hyperparameter optimization processes.
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
from pathlib import Path
from collections import defaultdict, deque
import warnings

from src.utils.tprint import tprint
from src.utils.logger import get_logger
from .enhanced_error_detector import detect_error, ErrorCategory, ErrorSeverity

logger = get_logger("EnhancedHPOMonitor")

class HPOStatus(Enum):
    """HPO optimization status."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    TIMEOUT = "timeout"
    CONVERGED = "converged"

class ConvergenceCriteria(Enum):
    """Convergence criteria types."""
    IMPROVEMENT_THRESHOLD = "improvement_threshold"
    PATIENCE = "patience"
    VARIANCE_THRESHOLD = "variance_threshold"
    CONFIDENCE_INTERVAL = "confidence_interval"
    BAYESIAN_CONVERGENCE = "bayesian_convergence"

@dataclass
class TrialResult:
    """Result of a single HPO trial."""
    trial_number: int
    timestamp: datetime
    parameters: Dict[str, Any]
    objective_value: float
    objective_std: Optional[float] = None
    training_time: Optional[float] = None
    memory_usage: Optional[float] = None
    convergence_info: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConvergenceInfo:
    """Convergence analysis information."""
    is_converged: bool
    convergence_criteria: List[ConvergenceCriteria]
    convergence_confidence: float
    improvement_rate: float
    variance_estimate: float
    best_value_history: List[float]
    convergence_analysis: Dict[str, Any]

@dataclass
class HPOStudyInfo:
    """Information about an HPO study."""
    study_id: str
    study_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: HPOStatus = HPOStatus.RUNNING
    total_trials: int = 0
    successful_trials: int = 0
    failed_trials: int = 0
    best_value: Optional[float] = None
    best_parameters: Optional[Dict[str, Any]] = None
    convergence_info: Optional[ConvergenceInfo] = None
    error_summary: Dict[str, int] = field(default_factory=dict)

class EnhancedHPOMonitor:
    """Enhanced HPO monitoring and failure detection system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced HPO monitor."""
        self.config = config or {}
        self.logger = logger.getChild('EnhancedHPOMonitor')
        
        # Study tracking
        self.active_studies: Dict[str, HPOStudyInfo] = {}
        self.completed_studies: Dict[str, HPOStudyInfo] = {}
        self.trial_results: Dict[str, List[TrialResult]] = defaultdict(list)
        
        # Monitoring configuration
        self.convergence_config = self.config.get('convergence', {
            'improvement_threshold': 0.001,
            'patience_trials': 20,
            'variance_threshold': 0.01,
            'confidence_level': 0.95,
            'min_trials_for_convergence': 10
        })
        
        self.failure_detection_config = self.config.get('failure_detection', {
            'max_failure_rate': 0.3,
            'consecutive_failures_threshold': 5,
            'timeout_threshold': 3600,  # 1 hour
            'memory_threshold': 0.9,  # 90% of available memory
            'performance_degradation_threshold': 0.1
        })
        
        # Early stopping configuration
        self.early_stopping_config = self.config.get('early_stopping', {
            'enable_early_stopping': True,
            'patience': 15,
            'min_delta': 0.001,
            'restore_best_weights': True,
            'monitor': 'objective_value'
        })
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.resource_usage_history: deque = deque(maxlen=1000)
        
        self.logger.info("🔍 Enhanced HPO Monitor initialized")
    
    def start_study(self, 
                   study_id: str, 
                   study_name: str,
                   search_space: Dict[str, Any],
                   objective_function: str) -> HPOStudyInfo:
        """Start monitoring a new HPO study."""
        try:
            study_info = HPOStudyInfo(
                study_id=study_id,
                study_name=study_name,
                start_time=datetime.now(),
                status=HPOStatus.RUNNING
            )
            
            with self.lock:
                self.active_studies[study_id] = study_info
                self.trial_results[study_id] = []
            
            self.logger.info(f"🚀 Started monitoring HPO study: {study_name} ({study_id})")
            return study_info
            
        except Exception as e:
            error_context = {
                'component': 'hpo_monitor',
                'function': 'start_study',
                'study_id': study_id,
                'study_name': study_name
            }
            detect_error(e, error_context)
            raise
    
    def record_trial_result(self, 
                          study_id: str,
                          trial_number: int,
                          parameters: Dict[str, Any],
                          objective_value: float,
                          **kwargs) -> TrialResult:
        """Record the result of a single trial."""
        try:
            trial_result = TrialResult(
                trial_number=trial_number,
                timestamp=datetime.now(),
                parameters=parameters,
                objective_value=objective_value,
                objective_std=kwargs.get('objective_std'),
                training_time=kwargs.get('training_time'),
                memory_usage=kwargs.get('memory_usage'),
                convergence_info=kwargs.get('convergence_info'),
                error_info=kwargs.get('error_info'),
                metadata=kwargs.get('metadata', {})
            )
            
            with self.lock:
                if study_id in self.active_studies:
                    self.trial_results[study_id].append(trial_result)
                    study_info = self.active_studies[study_id]
                    study_info.total_trials += 1
                    
                    if trial_result.error_info is None:
                        study_info.successful_trials += 1
                        
                        # Update best value
                        if (study_info.best_value is None or 
                            objective_value > study_info.best_value):
                            study_info.best_value = objective_value
                            study_info.best_parameters = parameters.copy()
                    else:
                        study_info.failed_trials += 1
                        self._update_error_summary(study_info, trial_result.error_info)
                    
                    # Check for convergence
                    convergence_info = self._check_convergence(study_id)
                    if convergence_info and convergence_info.is_converged:
                        study_info.convergence_info = convergence_info
                        study_info.status = HPOStatus.CONVERGED
                        self.logger.info(f"✅ Study {study_id} converged after {trial_number} trials")
                    
                    # Check for early stopping
                    if self._should_early_stop(study_id):
                        study_info.status = HPOStatus.STOPPED
                        self.logger.info(f"⏹️ Early stopping triggered for study {study_id}")
                    
                    # Check for failure conditions
                    if self._check_failure_conditions(study_id):
                        study_info.status = HPOStatus.FAILED
                        self.logger.error(f"❌ Study {study_id} failed due to failure conditions")
            
            # Track performance
            self._track_performance(study_id, trial_result)
            
            return trial_result
            
        except Exception as e:
            error_context = {
                'component': 'hpo_monitor',
                'function': 'record_trial_result',
                'study_id': study_id,
                'trial_number': trial_number
            }
            detect_error(e, error_context)
            raise
    
    def _check_convergence(self, study_id: str) -> Optional[ConvergenceInfo]:
        """Check if the study has converged."""
        try:
            if study_id not in self.trial_results:
                return None
            
            trial_results = self.trial_results[study_id]
            if len(trial_results) < self.convergence_config['min_trials_for_convergence']:
                return None
            
            # Extract objective values
            objective_values = [t.objective_value for t in trial_results if t.error_info is None]
            if len(objective_values) < self.convergence_config['min_trials_for_convergence']:
                return None
            
            convergence_criteria = []
            convergence_confidence = 0.0
            
            # Check improvement threshold
            if len(objective_values) >= 2:
                recent_improvement = abs(objective_values[-1] - objective_values[-2])
                if recent_improvement < self.convergence_config['improvement_threshold']:
                    convergence_criteria.append(ConvergenceCriteria.IMPROVEMENT_THRESHOLD)
                    convergence_confidence += 0.3
            
            # Check patience (no improvement for N trials)
            patience_trials = self.convergence_config['patience_trials']
            if len(objective_values) >= patience_trials:
                best_value = max(objective_values)
                recent_values = objective_values[-patience_trials:]
                if all(v <= best_value + self.convergence_config['improvement_threshold'] for v in recent_values):
                    convergence_criteria.append(ConvergenceCriteria.PATIENCE)
                    convergence_confidence += 0.4
            
            # Check variance threshold
            if len(objective_values) >= 10:
                recent_values = objective_values[-10:]
                variance = np.var(recent_values)
                if variance < self.convergence_config['variance_threshold']:
                    convergence_criteria.append(ConvergenceCriteria.VARIANCE_THRESHOLD)
                    convergence_confidence += 0.3
            
            # Calculate improvement rate
            if len(objective_values) >= 2:
                improvement_rate = (objective_values[-1] - objective_values[0]) / len(objective_values)
            else:
                improvement_rate = 0.0
            
            # Calculate variance estimate
            variance_estimate = np.var(objective_values) if len(objective_values) > 1 else 0.0
            
            # Determine if converged
            is_converged = (len(convergence_criteria) >= 2 and 
                          convergence_confidence >= 0.6)
            
            convergence_analysis = {
                'objective_values': objective_values,
                'recent_improvement': recent_improvement if len(objective_values) >= 2 else 0.0,
                'variance': variance_estimate,
                'improvement_rate': improvement_rate,
                'convergence_criteria_met': [c.value for c in convergence_criteria]
            }
            
            return ConvergenceInfo(
                is_converged=is_converged,
                convergence_criteria=convergence_criteria,
                convergence_confidence=convergence_confidence,
                improvement_rate=improvement_rate,
                variance_estimate=variance_estimate,
                best_value_history=objective_values,
                convergence_analysis=convergence_analysis
            )
            
        except Exception as e:
            error_context = {
                'component': 'hpo_monitor',
                'function': '_check_convergence',
                'study_id': study_id
            }
            detect_error(e, error_context)
            return None
    
    def _should_early_stop(self, study_id: str) -> bool:
        """Check if early stopping should be triggered."""
        try:
            if not self.early_stopping_config['enable_early_stopping']:
                return False
            
            if study_id not in self.trial_results:
                return False
            
            trial_results = self.trial_results[study_id]
            if len(trial_results) < self.early_stopping_config['patience']:
                return False
            
            # Check patience-based early stopping
            patience = self.early_stopping_config['patience']
            min_delta = self.early_stopping_config['min_delta']
            
            objective_values = [t.objective_value for t in trial_results if t.error_info is None]
            if len(objective_values) < patience:
                return False
            
            best_value = max(objective_values)
            recent_values = objective_values[-patience:]
            
            # Check if no improvement for patience trials
            if all(v <= best_value - min_delta for v in recent_values):
                return True
            
            return False
            
        except Exception as e:
            error_context = {
                'component': 'hpo_monitor',
                'function': '_should_early_stop',
                'study_id': study_id
            }
            detect_error(e, error_context)
            return False
    
    def _check_failure_conditions(self, study_id: str) -> bool:
        """Check if failure conditions are met."""
        try:
            if study_id not in self.active_studies:
                return False
            
            study_info = self.active_studies[study_id]
            trial_results = self.trial_results[study_id]
            
            # Check failure rate
            if study_info.total_trials > 0:
                failure_rate = study_info.failed_trials / study_info.total_trials
                if failure_rate > self.failure_detection_config['max_failure_rate']:
                    self.logger.error(f"❌ High failure rate: {failure_rate:.2%}")
                    return True
            
            # Check consecutive failures
            if len(trial_results) >= self.failure_detection_config['consecutive_failures_threshold']:
                recent_trials = trial_results[-self.failure_detection_config['consecutive_failures_threshold']:]
                if all(t.error_info is not None for t in recent_trials):
                    self.logger.error("❌ Too many consecutive failures")
                    return True
            
            # Check timeout
            if study_info.start_time:
                elapsed_time = (datetime.now() - study_info.start_time).total_seconds()
                if elapsed_time > self.failure_detection_config['timeout_threshold']:
                    self.logger.error(f"❌ Study timeout: {elapsed_time:.0f}s")
                    return True
            
            # Check memory usage
            if trial_results:
                recent_memory = [t.memory_usage for t in trial_results[-5:] if t.memory_usage is not None]
                if recent_memory:
                    avg_memory = np.mean(recent_memory)
                    if avg_memory > self.failure_detection_config['memory_threshold']:
                        self.logger.error(f"❌ High memory usage: {avg_memory:.2%}")
                        return True
            
            return False
            
        except Exception as e:
            error_context = {
                'component': 'hpo_monitor',
                'function': '_check_failure_conditions',
                'study_id': study_id
            }
            detect_error(e, error_context)
            return False
    
    def _update_error_summary(self, study_info: HPOStudyInfo, error_info: Dict[str, Any]):
        """Update error summary for a study."""
        try:
            error_type = error_info.get('error_type', 'unknown')
            study_info.error_summary[error_type] = study_info.error_summary.get(error_type, 0) + 1
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update error summary: {e}")
    
    def _track_performance(self, study_id: str, trial_result: TrialResult):
        """Track performance metrics."""
        try:
            performance_data = {
                'study_id': study_id,
                'trial_number': trial_result.trial_number,
                'timestamp': trial_result.timestamp,
                'objective_value': trial_result.objective_value,
                'training_time': trial_result.training_time,
                'memory_usage': trial_result.memory_usage,
                'success': trial_result.error_info is None
            }
            
            self.performance_history.append(performance_data)
            
            if trial_result.memory_usage is not None:
                resource_data = {
                    'study_id': study_id,
                    'timestamp': trial_result.timestamp,
                    'memory_usage': trial_result.memory_usage,
                    'training_time': trial_result.training_time
                }
                self.resource_usage_history.append(resource_data)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to track performance: {e}")
    
    def complete_study(self, study_id: str, final_status: HPOStatus = HPOStatus.COMPLETED):
        """Mark a study as completed."""
        try:
            with self.lock:
                if study_id in self.active_studies:
                    study_info = self.active_studies[study_id]
                    study_info.end_time = datetime.now()
                    study_info.status = final_status
                    
                    # Move to completed studies
                    self.completed_studies[study_id] = study_info
                    del self.active_studies[study_id]
                    
                    self.logger.info(f"✅ Study {study_id} completed with status: {final_status.value}")
                    
                    # Generate final report
                    self._generate_study_report(study_id)
            
        except Exception as e:
            error_context = {
                'component': 'hpo_monitor',
                'function': 'complete_study',
                'study_id': study_id
            }
            detect_error(e, error_context)
            raise
    
    def _generate_study_report(self, study_id: str):
        """Generate comprehensive study report."""
        try:
            if study_id not in self.completed_studies:
                return
            
            study_info = self.completed_studies[study_id]
            trial_results = self.trial_results[study_id]
            
            # Calculate statistics
            successful_trials = [t for t in trial_results if t.error_info is None]
            failed_trials = [t for t in trial_results if t.error_info is not None]
            
            if successful_trials:
                objective_values = [t.objective_value for t in successful_trials]
                training_times = [t.training_time for t in successful_trials if t.training_time is not None]
                memory_usages = [t.memory_usage for t in successful_trials if t.memory_usage is not None]
                
                stats = {
                    'best_objective_value': max(objective_values),
                    'worst_objective_value': min(objective_values),
                    'mean_objective_value': np.mean(objective_values),
                    'std_objective_value': np.std(objective_values),
                    'mean_training_time': np.mean(training_times) if training_times else None,
                    'mean_memory_usage': np.mean(memory_usages) if memory_usages else None,
                    'total_training_time': sum(training_times) if training_times else None
                }
            else:
                stats = {}
            
            # Create report
            report = {
                'study_info': {
                    'study_id': study_info.study_id,
                    'study_name': study_info.study_name,
                    'start_time': study_info.start_time.isoformat(),
                    'end_time': study_info.end_time.isoformat() if study_info.end_time else None,
                    'status': study_info.status.value,
                    'total_trials': study_info.total_trials,
                    'successful_trials': study_info.successful_trials,
                    'failed_trials': study_info.failed_trials,
                    'success_rate': study_info.successful_trials / max(1, study_info.total_trials)
                },
                'statistics': stats,
                'convergence_info': {
                    'is_converged': study_info.convergence_info.is_converged if study_info.convergence_info else False,
                    'convergence_criteria': [c.value for c in study_info.convergence_info.convergence_criteria] if study_info.convergence_info else [],
                    'convergence_confidence': study_info.convergence_info.convergence_confidence if study_info.convergence_info else 0.0
                } if study_info.convergence_info else None,
                'error_summary': study_info.error_summary,
                'best_parameters': study_info.best_parameters,
                'trial_count': len(trial_results)
            }
            
            # Save report
            report_dir = Path("hpo_reports")
            report_dir.mkdir(exist_ok=True)
            report_file = report_dir / f"study_report_{study_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"📊 Study report saved: {report_file}")
            
        except Exception as e:
            error_context = {
                'component': 'hpo_monitor',
                'function': '_generate_study_report',
                'study_id': study_id
            }
            detect_error(e, error_context)
    
    def get_study_status(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a study."""
        try:
            with self.lock:
                study_info = self.active_studies.get(study_id) or self.completed_studies.get(study_id)
                if not study_info:
                    return None
                
                trial_results = self.trial_results.get(study_id, [])
                
                return {
                    'study_id': study_info.study_id,
                    'study_name': study_info.study_name,
                    'status': study_info.status.value,
                    'start_time': study_info.start_time.isoformat(),
                    'end_time': study_info.end_time.isoformat() if study_info.end_time else None,
                    'total_trials': study_info.total_trials,
                    'successful_trials': study_info.successful_trials,
                    'failed_trials': study_info.failed_trials,
                    'best_value': study_info.best_value,
                    'convergence_info': {
                        'is_converged': study_info.convergence_info.is_converged if study_info.convergence_info else False,
                        'convergence_confidence': study_info.convergence_info.convergence_confidence if study_info.convergence_info else 0.0
                    } if study_info.convergence_info else None,
                    'error_summary': study_info.error_summary,
                    'recent_trials': len(trial_results[-10:]) if trial_results else 0
                }
                
        except Exception as e:
            error_context = {
                'component': 'hpo_monitor',
                'function': 'get_study_status',
                'study_id': study_id
            }
            detect_error(e, error_context)
            return None
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        try:
            with self.lock:
                active_count = len(self.active_studies)
                completed_count = len(self.completed_studies)
                
                # Calculate overall statistics
                all_trials = []
                for study_id in self.trial_results:
                    all_trials.extend(self.trial_results[study_id])
                
                successful_trials = [t for t in all_trials if t.error_info is None]
                failed_trials = [t for t in all_trials if t.error_info is not None]
                
                total_trials = len(all_trials)
                success_rate = len(successful_trials) / max(1, total_trials)
                
                # Calculate performance metrics
                if successful_trials:
                    objective_values = [t.objective_value for t in successful_trials]
                    training_times = [t.training_time for t in successful_trials if t.training_time is not None]
                    
                    performance_metrics = {
                        'best_objective_value': max(objective_values),
                        'mean_objective_value': np.mean(objective_values),
                        'mean_training_time': np.mean(training_times) if training_times else None,
                        'total_training_time': sum(training_times) if training_times else None
                    }
                else:
                    performance_metrics = {}
                
                # Error analysis
                error_types = defaultdict(int)
                for trial in failed_trials:
                    if trial.error_info:
                        error_type = trial.error_info.get('error_type', 'unknown')
                        error_types[error_type] += 1
                
                return {
                    'monitoring_summary': {
                        'active_studies': active_count,
                        'completed_studies': completed_count,
                        'total_trials': total_trials,
                        'successful_trials': len(successful_trials),
                        'failed_trials': len(failed_trials),
                        'overall_success_rate': success_rate,
                        'monitoring_active': self.monitoring_active
                    },
                    'performance_metrics': performance_metrics,
                    'error_analysis': dict(error_types),
                    'convergence_analysis': {
                        'converged_studies': sum(1 for s in self.completed_studies.values() 
                                               if s.convergence_info and s.convergence_info.is_converged),
                        'early_stopped_studies': sum(1 for s in self.completed_studies.values() 
                                                   if s.status == HPOStatus.STOPPED)
                    }
                }
                
        except Exception as e:
            error_context = {
                'component': 'hpo_monitor',
                'function': 'get_monitoring_summary'
            }
            detect_error(e, error_context)
            return {'error': str(e)}
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("🔍 HPO monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("🔍 HPO monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                with self.lock:
                    # Check for timeout studies
                    current_time = datetime.now()
                    for study_id, study_info in list(self.active_studies.items()):
                        if study_info.start_time:
                            elapsed_time = (current_time - study_info.start_time).total_seconds()
                            if elapsed_time > self.failure_detection_config['timeout_threshold']:
                                study_info.status = HPOStatus.TIMEOUT
                                self.complete_study(study_id, HPOStatus.TIMEOUT)
                
                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring loop error: {e}")
                time.sleep(30)

# Global HPO monitor instance
_global_hpo_monitor = None

def get_global_hpo_monitor(config: Optional[Dict[str, Any]] = None) -> EnhancedHPOMonitor:
    """Get or create global HPO monitor instance."""
    global _global_hpo_monitor
    
    if _global_hpo_monitor is None:
        _global_hpo_monitor = EnhancedHPOMonitor(config)
        _global_hpo_monitor.start_monitoring()
    
    return _global_hpo_monitor