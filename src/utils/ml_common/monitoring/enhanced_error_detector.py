#!/usr/bin/env python3
"""
Enhanced Error Detection and Classification System for ML Pipelines

This module provides comprehensive error detection, classification, and handling
for ML training, HPO, and testing pipelines to prevent silent failures.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import traceback
import warnings
from pathlib import Path
import json
import pickle
from collections import defaultdict, deque
import threading
import time

from src.utils.tprint import tprint
from src.utils.logger import get_logger

logger = get_logger("EnhancedErrorDetector")

class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"      # Pipeline should stop immediately
    HIGH = "high"             # Significant issue, may affect results
    MEDIUM = "medium"         # Warning, should be investigated
    LOW = "low"              # Minor issue, can continue
    INFO = "info"            # Informational

class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA_QUALITY = "data_quality"
    MODEL_TRAINING = "model_training"
    HPO_OPTIMIZATION = "hpo_optimization"
    VALIDATION = "validation"
    MEMORY = "memory"
    PERFORMANCE = "performance"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    TIMEOUT = "timeout"
    CONVERGENCE = "convergence"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    FEATURE_ENGINEERING = "feature_engineering"
    CROSS_VALIDATION = "cross_validation"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for an error."""
    timestamp: datetime
    component: str
    function: str
    line_number: int
    error_type: str
    error_message: str
    stack_trace: str
    input_data_shape: Optional[Tuple] = None
    input_data_dtypes: Optional[Dict] = None
    memory_usage: Optional[float] = None
    execution_time: Optional[float] = None
    model_type: Optional[str] = None
    hyperparameters: Optional[Dict] = None
    data_characteristics: Optional[Dict] = None

@dataclass
class ErrorRecord:
    """Complete error record with classification and context."""
    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    classification_confidence: float
    suggested_actions: List[str]
    related_errors: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_notes: Optional[str] = None

class EnhancedErrorDetector:
    """Enhanced error detection and classification system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced error detector."""
        self.config = config or {}
        self.logger = logger.getChild('EnhancedErrorDetector')

        # Error tracking
        self.error_history: deque = deque(maxlen=1000)
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.component_errors: Dict[str, List[ErrorRecord]] = defaultdict(list)

        # Configuration
        self.enable_real_time_monitoring = self.config.get('enable_real_time_monitoring', True)
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'critical_errors_per_hour': 5,
            'high_errors_per_hour': 20,
            'same_error_repetition': 10,
            'component_failure_rate': 0.3
        })

        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()

        # Error classification rules
        self.classification_rules = self._initialize_classification_rules()

        self.logger.info("🔍 Enhanced Error Detector initialized")

    def _initialize_classification_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error classification rules."""
        return {
            'data_quality': {
                'patterns': [
                    r'single class',
                    r'class imbalance',
                    r'empty data',
                    r'nan values',
                    r'infinite values',
                    r'data type mismatch',
                    r'schema validation'
                ],
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.DATA_QUALITY,
                'confidence_threshold': 0.8
            },
            'model_training': {
                'patterns': [
                    r'fit failed',
                    r'training error',
                    r'convergence failed',
                    r'gradient explosion',
                    r'gradient vanishing',
                    r'loss nan',
                    r'loss infinite'
                ],
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.MODEL_TRAINING,
                'confidence_threshold': 0.8
            },
            'hpo_optimization': {
                'patterns': [
                    r'optuna error',
                    r'hyperparameter',
                    r'optimization failed',
                    r'trial failed',
                    r'study error',
                    r'pruning error'
                ],
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.HPO_OPTIMIZATION,
                'confidence_threshold': 0.7
            },
            'memory': {
                'patterns': [
                    r'out of memory',
                    r'memory error',
                    r'oom',
                    r'memory allocation',
                    r'cuda out of memory'
                ],
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.MEMORY,
                'confidence_threshold': 0.9
            },
            'timeout': {
                'patterns': [
                    r'timeout',
                    r'took too long',
                    r'exceeded time limit',
                    r'deadline exceeded'
                ],
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.TIMEOUT,
                'confidence_threshold': 0.8
            },
            'convergence': {
                'patterns': [
                    r'not converged',
                    r'convergence warning',
                    r'max iterations',
                    r'early stopping'
                ],
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.CONVERGENCE,
                'confidence_threshold': 0.7
            },
            'overfitting': {
                'patterns': [
                    r'overfitting',
                    r'validation loss increasing',
                    r'gap between train and val',
                    r'high variance'
                ],
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.OVERFITTING,
                'confidence_threshold': 0.6
            },
            'underfitting': {
                'patterns': [
                    r'underfitting',
                    r'high bias',
                    r'low performance',
                    r'poor fit'
                ],
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.UNDERFITTING,
                'confidence_threshold': 0.6
            }
        }

    def detect_and_classify_error(self,
                                error: Exception,
                                context: Dict[str, Any]) -> ErrorRecord:
        """Detect and classify an error with comprehensive analysis."""
        try:
            # Create error context
            error_context = self._create_error_context(error, context)

            # Classify the error
            classification = self._classify_error(error, error_context)

            # Generate error ID
            error_id = self._generate_error_id(error_context)

            # Create error record
            error_record = ErrorRecord(
                error_id=error_id,
                severity=classification['severity'],
                category=classification['category'],
                context=error_context,
                classification_confidence=classification['confidence'],
                suggested_actions=classification['suggested_actions']
            )

            # Store error record
            with self.lock:
                self.error_history.append(error_record)
                self.component_errors[error_context.component].append(error_record)
                self.error_patterns[error_record.error_id] += 1

            # Check for alert conditions
            self._check_alert_conditions(error_record)

            # Log error
            self._log_error(error_record)

            return error_record

        except Exception as e:
            self.logger.error(f"❌ Error detection failed: {e}")
            # Return a fallback error record
            return self._create_fallback_error_record(error, context)

    def _create_error_context(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Create comprehensive error context."""
        try:
            # Extract stack trace information
            tb = traceback.extract_tb(error.__traceback__)
            frame = tb[-1] if tb else None

            return ErrorContext(
                timestamp=datetime.now(),
                component=context.get('component', 'unknown'),
                function=frame.name if frame else 'unknown',
                line_number=frame.lineno if frame else 0,
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                input_data_shape=context.get('input_data_shape'),
                input_data_dtypes=context.get('input_data_dtypes'),
                memory_usage=context.get('memory_usage'),
                execution_time=context.get('execution_time'),
                model_type=context.get('model_type'),
                hyperparameters=context.get('hyperparameters'),
                data_characteristics=context.get('data_characteristics')
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create error context: {e}")
            return ErrorContext(
                timestamp=datetime.now(),
                component='unknown',
                function='unknown',
                line_number=0,
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc()
            )

    def _classify_error(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Classify error based on patterns and context."""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()

        best_match = None
        best_confidence = 0.0

        # Check against classification rules
        for rule_name, rule_config in self.classification_rules.items():
            confidence = self._calculate_classification_confidence(
                error_message, error_type, context, rule_config
            )

            if confidence > best_confidence and confidence >= rule_config['confidence_threshold']:
                best_confidence = confidence
                best_match = rule_config

        # Default classification if no match found
        if best_match is None:
            best_match = {
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.UNKNOWN,
                'confidence_threshold': 0.5
            }
            best_confidence = 0.5

        # Generate suggested actions
        suggested_actions = self._generate_suggested_actions(
            best_match['category'], context, best_confidence
        )

        return {
            'severity': best_match['severity'],
            'category': best_match['category'],
            'confidence': best_confidence,
            'suggested_actions': suggested_actions
        }

    def _calculate_classification_confidence(self,
                                           error_message: str,
                                           error_type: str,
                                           context: ErrorContext,
                                           rule_config: Dict[str, Any]) -> float:
        """Calculate confidence score for error classification."""
        import re

        confidence = 0.0
        patterns = rule_config['patterns']

        # Pattern matching
        for pattern in patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                confidence += 0.3

        # Error type matching
        if rule_config['category'].value.replace('_', '') in error_type:
            confidence += 0.2

        # Context-based scoring
        if context.component and rule_config['category'].value in context.component.lower():
            confidence += 0.2

        if context.model_type and rule_config['category'].value in context.model_type.lower():
            confidence += 0.1

        # Data characteristics scoring
        if context.data_characteristics:
            if rule_config['category'] == ErrorCategory.DATA_QUALITY:
                if context.data_characteristics.get('has_nan', False):
                    confidence += 0.1
                if context.data_characteristics.get('has_inf', False):
                    confidence += 0.1
                if context.data_characteristics.get('single_class', False):
                    confidence += 0.2

        return min(confidence, 1.0)

    def _generate_suggested_actions(self,
                                  category: ErrorCategory,
                                  context: ErrorContext,
                                  confidence: float) -> List[str]:
        """Generate suggested actions based on error category."""
        actions = []

        if category == ErrorCategory.DATA_QUALITY:
            actions.extend([
                "Check data preprocessing pipeline",
                "Validate input data schema",
                "Handle missing values appropriately",
                "Check for class imbalance",
                "Verify data types and ranges"
            ])

        elif category == ErrorCategory.MODEL_TRAINING:
            actions.extend([
                "Check model hyperparameters",
                "Verify training data quality",
                "Consider reducing model complexity",
                "Check for gradient issues",
                "Validate loss function"
            ])

        elif category == ErrorCategory.HPO_OPTIMIZATION:
            actions.extend([
                "Check HPO search space",
                "Verify objective function",
                "Consider reducing number of trials",
                "Check for memory issues during optimization",
                "Validate cross-validation setup"
            ])

        elif category == ErrorCategory.MEMORY:
            actions.extend([
                "Reduce batch size",
                "Use data streaming",
                "Clear unused variables",
                "Consider model quantization",
                "Check for memory leaks"
            ])

        elif category == ErrorCategory.TIMEOUT:
            actions.extend([
                "Increase timeout limits",
                "Optimize algorithm performance",
                "Reduce data size for testing",
                "Use early stopping",
                "Check for infinite loops"
            ])

        elif category == ErrorCategory.CONVERGENCE:
            actions.extend([
                "Adjust learning rate",
                "Increase maximum iterations",
                "Check for numerical stability",
                "Consider different optimizer",
                "Validate convergence criteria"
            ])

        elif category == ErrorCategory.OVERFITTING:
            actions.extend([
                "Increase regularization",
                "Reduce model complexity",
                "Add more training data",
                "Use early stopping",
                "Implement cross-validation"
            ])

        elif category == ErrorCategory.UNDERFITTING:
            actions.extend([
                "Increase model complexity",
                "Reduce regularization",
                "Check feature engineering",
                "Increase training time",
                "Verify data quality"
            ])

        else:
            actions.extend([
                "Review error logs",
                "Check system resources",
                "Validate configuration",
                "Contact support if persistent"
            ])

        # Add confidence-based actions
        if confidence < 0.7:
            actions.append("Manual review recommended - low classification confidence")

        return actions[:5]  # Limit to top 5 actions

    def _generate_error_id(self, context: ErrorContext) -> str:
        """Generate unique error ID based on context."""
        import hashlib

        # Create hash from key context elements
        key_elements = [
            context.component,
            context.function,
            context.error_type,
            context.error_message[:100]  # First 100 chars of message
        ]

        key_string = "|".join(str(elem) for elem in key_elements)
        error_hash = hashlib.md5(key_string.encode()).hexdigest()[:8]

        return f"{context.component}_{context.error_type}_{error_hash}"

    def _check_alert_conditions(self, error_record: ErrorRecord):
        """Check if error conditions warrant alerts."""
        try:
            current_time = datetime.now()
            one_hour_ago = current_time - timedelta(hours=1)

            # Count recent errors by severity
            recent_errors = [
                err for err in self.error_history
                if err.context.timestamp > one_hour_ago
            ]

            critical_count = sum(1 for err in recent_errors if err.severity == ErrorSeverity.CRITICAL)
            high_count = sum(1 for err in recent_errors if err.severity == ErrorSeverity.HIGH)

            # Check thresholds
            if critical_count >= self.alert_thresholds['critical_errors_per_hour']:
                self._trigger_alert("CRITICAL", f"Too many critical errors: {critical_count}")

            if high_count >= self.alert_thresholds['high_errors_per_hour']:
                self._trigger_alert("HIGH", f"Too many high severity errors: {high_count}")

            # Check for repeated errors
            if self.error_patterns[error_record.error_id] >= self.alert_thresholds['same_error_repetition']:
                self._trigger_alert("REPETITION", f"Error repeated {self.error_patterns[error_record.error_id]} times: {error_record.error_id}")

            # Check component failure rate
            component_errors = self.component_errors[error_record.context.component]
            if len(component_errors) > 10:  # Only check if we have enough data
                recent_component_errors = [
                    err for err in component_errors
                    if err.context.timestamp > one_hour_ago
                ]
                failure_rate = len(recent_component_errors) / max(1, len(component_errors))

                if failure_rate >= self.alert_thresholds['component_failure_rate']:
                    self._trigger_alert("COMPONENT", f"High failure rate in {error_record.context.component}: {failure_rate:.2%}")

        except Exception as e:
            self.logger.error(f"❌ Alert condition check failed: {e}")

    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger an alert for critical conditions."""
        alert_message = f"🚨 ALERT [{alert_type}]: {message}"

        # Log alert
        self.logger.critical(alert_message)
        tprint(alert_message)

        # In a real implementation, you might:
        # - Send email notifications
        # - Post to Slack/Teams
        # - Create tickets
        # - Send SMS alerts

        # For now, we'll save to a file
        try:
            alert_file = Path("alerts") / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            alert_file.parent.mkdir(exist_ok=True)

            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': alert_type,
                'message': message,
                'error_count': len(self.error_history),
                'component_errors': dict(self.component_errors)
            }

            with open(alert_file, 'w') as f:
                json.dump(alert_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"❌ Failed to save alert: {e}")

    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level."""
        log_message = (
            f"Error [{error_record.severity.value.upper()}] "
            f"[{error_record.category.value}] "
            f"[{error_record.context.component}] "
            f"{error_record.context.error_message[:100]}..."
        )

        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def _create_fallback_error_record(self, error: Exception, context: Dict[str, Any]) -> ErrorRecord:
        """Create a fallback error record when detection fails."""
        fallback_context = ErrorContext(
            timestamp=datetime.now(),
            component=context.get('component', 'unknown'),
            function='unknown',
            line_number=0,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc()
        )

        return ErrorRecord(
            error_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.UNKNOWN,
            context=fallback_context,
            classification_confidence=0.0,
            suggested_actions=["Manual investigation required"]
        )

    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""
        with self.lock:
            current_time = datetime.now()
            one_hour_ago = current_time - timedelta(hours=1)
            one_day_ago = current_time - timedelta(days=1)

            recent_errors = [err for err in self.error_history if err.context.timestamp > one_hour_ago]
            daily_errors = [err for err in self.error_history if err.context.timestamp > one_day_ago]

            # Count by severity
            severity_counts = defaultdict(int)
            for err in self.error_history:
                severity_counts[err.severity.value] += 1

            # Count by category
            category_counts = defaultdict(int)
            for err in self.error_history:
                category_counts[err.category.value] += 1

            # Count by component
            component_counts = defaultdict(int)
            for err in self.error_history:
                component_counts[err.context.component] += 1

            # Most frequent errors
            most_frequent = sorted(
                self.error_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            return {
                'total_errors': len(self.error_history),
                'recent_errors_1h': len(recent_errors),
                'daily_errors': len(daily_errors),
                'severity_distribution': dict(severity_counts),
                'category_distribution': dict(category_counts),
                'component_distribution': dict(component_counts),
                'most_frequent_errors': most_frequent,
                'unresolved_errors': sum(1 for err in self.error_history if not err.resolved),
                'monitoring_active': self.monitoring_active
            }

    def start_monitoring(self):
        """Start real-time error monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("🔍 Real-time error monitoring started")

    def stop_monitoring(self):
        """Stop real-time error monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self.logger.info("🔍 Real-time error monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check for alert conditions
                self._check_alert_conditions(ErrorRecord(
                    error_id="monitor",
                    severity=ErrorSeverity.INFO,
                    category=ErrorCategory.UNKNOWN,
                    context=ErrorContext(
                        timestamp=datetime.now(),
                        component="monitor",
                        function="monitoring_loop",
                        line_number=0,
                        error_type="Monitor",
                        error_message="Monitoring check",
                        stack_trace=""
                    ),
                    classification_confidence=0.0,
                    suggested_actions=[]
                ))

                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"❌ Monitoring loop error: {e}")
                time.sleep(60)

    def save_error_report(self, filepath: str):
        """Save comprehensive error report."""
        try:
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_errors': len(self.error_history),
                    'monitoring_duration': 'unknown'  # Could track this
                },
                'error_summary': self.get_error_summary(),
                'error_details': [
                    {
                        'error_id': err.error_id,
                        'timestamp': err.context.timestamp.isoformat(),
                        'severity': err.severity.value,
                        'category': err.category.value,
                        'component': err.context.component,
                        'function': err.context.function,
                        'error_type': err.context.error_type,
                        'error_message': err.context.error_message,
                        'confidence': err.classification_confidence,
                        'suggested_actions': err.suggested_actions,
                        'resolved': err.resolved
                    }
                    for err in self.error_history
                ]
            }

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"✅ Error report saved to: {filepath}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save error report: {e}")

# Global error detector instance
_global_error_detector = None

def get_global_error_detector(config: Optional[Dict[str, Any]] = None) -> EnhancedErrorDetector:
    """Get or create global error detector instance."""
    global _global_error_detector

    if _global_error_detector is None:
        _global_error_detector = EnhancedErrorDetector(config)
        _global_error_detector.start_monitoring()

    return _global_error_detector

def detect_error(error: Exception, context: Dict[str, Any]) -> ErrorRecord:
    """Convenience function to detect and classify an error."""
    detector = get_global_error_detector()
    return detector.detect_and_classify_error(error, context)

# Export aliases for backward compatibility
ErrorDetector = EnhancedErrorDetector
ErrorHandler = EnhancedErrorDetector  # The detector handles errors
ErrorReporter = EnhancedErrorDetector  # The detector also reports errors
