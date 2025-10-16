"""
Validation Reporting Integration

Integrates universal validation system with the enhanced reporting system
to provide comprehensive validation reports, alerts, and monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path

# Import validation components
from ..validation import (
    UniversalMLValidationReport,
    OverfittingReport,
    TemporalValidationReport,
    get_ml_validator,
    validate_ml_model
)

# Import reporting system
from .enhanced_reporting_system import (
    EnhancedReportingSystem,
    ReportData,
    ReportType,
    Alert,
    AlertLevel
)

logger = logging.getLogger(__name__)

@dataclass
class ValidationReportData:
    """Enhanced validation report data structure."""

    # Basic validation info
    model_name: str
    model_type: str
    validation_timestamp: datetime
    validation_score: float
    validation_passed: bool

    # Detailed validation results
    overfitting_analysis: Optional[Dict[str, Any]] = None
    temporal_validation: Optional[Dict[str, Any]] = None
    timeframe_validation: Optional[Dict[str, Any]] = None

    # Issues and recommendations
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    fold_number: Optional[int] = None
    trial_number: Optional[int] = None
    validation_duration: Optional[float] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)

class ValidationReportingIntegrator:
    """Integrates validation system with enhanced reporting."""

    def __init__(self, reporting_system: Optional[EnhancedReportingSystem] = None):
        """
        Initialize validation reporting integrator.

        Args:
            reporting_system: Enhanced reporting system instance
        """
        self.reporting_system = reporting_system or EnhancedReportingSystem()
        self.validation_reports: Dict[str, ValidationReportData] = {}
        self.validation_history: List[ValidationReportData] = []

        # Validation monitoring configuration
        self.monitoring_config = {
            'enable_validation_alerts': True,
            'validation_failure_threshold': 0.5,
            'overfitting_severity_threshold': 'high',
            'alert_on_critical_issues': True,
            'alert_on_validation_failures': True,
            'generate_validation_reports': True,
            'validation_report_directory': 'reports/validation'
        }

        # Create validation report directory
        Path(self.monitoring_config['validation_report_directory']).mkdir(parents=True, exist_ok=True)

        logger.info("✅ Validation Reporting Integrator initialized")

    def process_validation_report(self,
                                 validation_report: Union[UniversalMLValidationReport, Dict[str, Any]],
                                 model_name: str = "unknown",
                                 model_type: str = "unknown",
                                 fold_number: Optional[int] = None,
                                 trial_number: Optional[int] = None,
                                 validation_duration: Optional[float] = None,
                                 model_metadata: Optional[Dict[str, Any]] = None) -> ValidationReportData:
        """
        Process validation report and integrate with reporting system.

        Args:
            validation_report: Validation report to process
            model_name: Name of the model
            model_type: Type of model
            fold_number: Fold number for cross-validation
            trial_number: Trial number for HPO
            validation_duration: Duration of validation in seconds
            model_metadata: Additional model metadata

        Returns:
            ValidationReportData: Processed validation report data
        """
        try:
            # Convert to dict if needed
            if hasattr(validation_report, '__dict__'):
                validation_dict = validation_report.__dict__
            else:
                validation_dict = validation_report

            # Extract validation data
            validation_data = ValidationReportData(
                model_name=model_name,
                model_type=model_type,
                validation_timestamp=datetime.now(),
                validation_score=validation_dict.get('validation_score', 0.0),
                validation_passed=validation_dict.get('overall_validation_passed', False),
                overfitting_analysis=self._extract_overfitting_analysis(validation_dict),
                temporal_validation=self._extract_temporal_validation(validation_dict),
                timeframe_validation=self._extract_timeframe_validation(validation_dict),
                critical_issues=validation_dict.get('critical_issues', []),
                warnings=validation_dict.get('warnings', []),
                recommendations=validation_dict.get('recommendations', []),
                fold_number=fold_number,
                trial_number=trial_number,
                validation_duration=validation_duration,
                model_metadata=model_metadata or {}
            )

            # Store validation data
            self.validation_reports[f"{model_name}_{model_type}"] = validation_data
            self.validation_history.append(validation_data)

            # Generate comprehensive report
            self._generate_validation_report(validation_data)

            # Check for alerts
            self._check_validation_alerts(validation_data)

            # Update monitoring
            self._update_validation_monitoring(validation_data)

            return validation_data

        except Exception as e:
            logger.error(f"Failed to process validation report: {e}")
            return ValidationReportData(
                model_name=model_name,
                model_type=model_type,
                validation_timestamp=datetime.now(),
                validation_score=0.0,
                validation_passed=False,
                critical_issues=[f"Validation processing failed: {str(e)}"]
            )

    def _extract_overfitting_analysis(self, validation_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract overfitting analysis from validation report."""
        overfitting_analysis = validation_dict.get('overfitting_analysis')
        if overfitting_analysis is None:
            return None

        if hasattr(overfitting_analysis, '__dict__'):
            return overfitting_analysis.__dict__
        else:
            return overfitting_analysis

    def _extract_temporal_validation(self, validation_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract temporal validation from validation report."""
        temporal_validation = validation_dict.get('temporal_validation')
        if temporal_validation is None:
            return None

        if hasattr(temporal_validation, '__dict__'):
            return temporal_validation.__dict__
        else:
            return temporal_validation

    def _extract_timeframe_validation(self, validation_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract timeframe validation from validation report."""
        timeframe_validation = validation_dict.get('timeframe_validation')
        if timeframe_validation is None:
            return None

        return timeframe_validation

    def _generate_validation_report(self, validation_data: ValidationReportData):
        """Generate comprehensive validation report."""
        try:
            # Create report data
            report_data = ReportData(
                report_id=f"validation_{validation_data.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                report_type=ReportType.MODEL_VALIDATION,
                timestamp=validation_data.validation_timestamp,
                title=f"Validation Report: {validation_data.model_name}",
                summary=self._generate_validation_summary(validation_data),
                data={
                    'validation_score': validation_data.validation_score,
                    'validation_passed': validation_data.validation_passed,
                    'overfitting_analysis': validation_data.overfitting_analysis,
                    'temporal_validation': validation_data.temporal_validation,
                    'timeframe_validation': validation_data.timeframe_validation,
                    'critical_issues': validation_data.critical_issues,
                    'warnings': validation_data.warnings,
                    'recommendations': validation_data.recommendations,
                    'model_metadata': validation_data.model_metadata
                },
                metrics={
                    'validation_score': validation_data.validation_score,
                    'critical_issues_count': len(validation_data.critical_issues),
                    'warnings_count': len(validation_data.warnings),
                    'recommendations_count': len(validation_data.recommendations)
                },
                recommendations=validation_data.recommendations,
                metadata={
                    'model_name': validation_data.model_name,
                    'model_type': validation_data.model_type,
                    'fold_number': validation_data.fold_number,
                    'trial_number': validation_data.trial_number,
                    'validation_duration': validation_data.validation_duration
                }
            )

            # Add to reporting system
            self.reporting_system.add_report(report_data)

            # Save validation report to file
            self._save_validation_report(validation_data)

        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")

    def _generate_validation_summary(self, validation_data: ValidationReportData) -> str:
        """Generate validation summary."""
        status = "✅ PASSED" if validation_data.validation_passed else "❌ FAILED"
        score = validation_data.validation_score

        summary = f"Model {validation_data.model_name} validation {status} (Score: {score:.3f})"

        if validation_data.critical_issues:
            summary += f" - {len(validation_data.critical_issues)} critical issues"

        if validation_data.warnings:
            summary += f" - {len(validation_data.warnings)} warnings"

        if validation_data.overfitting_analysis:
            overfitting = validation_data.overfitting_analysis
            if overfitting.get('is_overfitting', False):
                severity = overfitting.get('severity', 'unknown')
                summary += f" - Overfitting detected ({severity})"

        return summary

    def _save_validation_report(self, validation_data: ValidationReportData):
        """Save validation report to file."""
        try:
            report_dict = {
                'model_name': validation_data.model_name,
                'model_type': validation_data.model_type,
                'validation_timestamp': validation_data.validation_timestamp.isoformat(),
                'validation_score': validation_data.validation_score,
                'validation_passed': validation_data.validation_passed,
                'overfitting_analysis': validation_data.overfitting_analysis,
                'temporal_validation': validation_data.temporal_validation,
                'timeframe_validation': validation_data.timeframe_validation,
                'critical_issues': validation_data.critical_issues,
                'warnings': validation_data.warnings,
                'recommendations': validation_data.recommendations,
                'fold_number': validation_data.fold_number,
                'trial_number': validation_data.trial_number,
                'validation_duration': validation_data.validation_duration,
                'model_metadata': validation_data.model_metadata
            }

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{validation_data.model_name}_{timestamp}.json"
            filepath = Path(self.monitoring_config['validation_report_directory']) / filename

            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)

            logger.info(f"Validation report saved: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")

    def _check_validation_alerts(self, validation_data: ValidationReportData):
        """Check for validation alerts."""
        try:
            # Check validation failure
            if not validation_data.validation_passed:
                alert = Alert(
                    alert_id=f"validation_failure_{validation_data.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.ERROR,
                    title=f"Validation Failed: {validation_data.model_name}",
                    message=f"Model {validation_data.model_name} validation failed with score {validation_data.validation_score:.3f}",
                    timestamp=datetime.now(),
                    component="validation_system",
                    data={
                        'model_name': validation_data.model_name,
                        'model_type': validation_data.model_type,
                        'validation_score': validation_data.validation_score,
                        'critical_issues': validation_data.critical_issues
                    }
                )
                self.reporting_system.add_alert(alert)

            # Check critical issues
            if validation_data.critical_issues:
                alert = Alert(
                    alert_id=f"critical_issues_{validation_data.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.CRITICAL,
                    title=f"Critical Issues: {validation_data.model_name}",
                    message=f"Model {validation_data.model_name} has {len(validation_data.critical_issues)} critical issues",
                    timestamp=datetime.now(),
                    component="validation_system",
                    data={
                        'model_name': validation_data.model_name,
                        'model_type': validation_data.model_type,
                        'critical_issues': validation_data.critical_issues
                    }
                )
                self.reporting_system.add_alert(alert)

            # Check overfitting
            if validation_data.overfitting_analysis:
                overfitting = validation_data.overfitting_analysis
                if overfitting.get('is_overfitting', False):
                    severity = overfitting.get('severity', 'unknown')
                    if severity in ['high', 'severe']:
                        alert = Alert(
                            alert_id=f"overfitting_{validation_data.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            level=AlertLevel.WARNING,
                            title=f"Overfitting Detected: {validation_data.model_name}",
                            message=f"Model {validation_data.model_name} shows {severity} overfitting",
                            timestamp=datetime.now(),
                            component="validation_system",
                            data={
                                'model_name': validation_data.model_name,
                                'model_type': validation_data.model_type,
                                'overfitting_analysis': overfitting
                            }
                        )
                        self.reporting_system.add_alert(alert)

            # Check low validation score
            if validation_data.validation_score < self.monitoring_config['validation_failure_threshold']:
                alert = Alert(
                    alert_id=f"low_score_{validation_data.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.WARNING,
                    title=f"Low Validation Score: {validation_data.model_name}",
                    message=f"Model {validation_data.model_name} has low validation score: {validation_data.validation_score:.3f}",
                    timestamp=datetime.now(),
                    component="validation_system",
                    data={
                        'model_name': validation_data.model_name,
                        'model_type': validation_data.model_type,
                        'validation_score': validation_data.validation_score
                    }
                )
                self.reporting_system.add_alert(alert)

        except Exception as e:
            logger.error(f"Failed to check validation alerts: {e}")

    def _update_validation_monitoring(self, validation_data: ValidationReportData):
        """Update validation monitoring."""
        try:
            # Update monitoring metrics
            monitoring_data = {
                'timestamp': validation_data.validation_timestamp,
                'model_name': validation_data.model_name,
                'model_type': validation_data.model_type,
                'validation_score': validation_data.validation_score,
                'validation_passed': validation_data.validation_passed,
                'critical_issues_count': len(validation_data.critical_issues),
                'warnings_count': len(validation_data.warnings),
                'overfitting_detected': validation_data.overfitting_analysis.get('is_overfitting', False) if validation_data.overfitting_analysis else False
            }

            # Add to monitoring system
            self.reporting_system.update_monitoring_metrics('validation', monitoring_data)

        except Exception as e:
            logger.error(f"Failed to update validation monitoring: {e}")

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        if not self.validation_history:
            return {'message': 'No validation reports available'}

        # Calculate summary statistics
        total_validations = len(self.validation_history)
        passed_validations = sum(1 for v in self.validation_history if v.validation_passed)
        success_rate = passed_validations / total_validations

        # Model type distribution
        model_type_counts = {}
        for validation in self.validation_history:
            model_type = validation.model_type
            model_type_counts[model_type] = model_type_counts.get(model_type, 0) + 1

        # Average validation scores
        avg_validation_score = np.mean([v.validation_score for v in self.validation_history])

        # Critical issues summary
        total_critical_issues = sum(len(v.critical_issues) for v in self.validation_history)
        total_warnings = sum(len(v.warnings) for v in self.validation_history)

        # Overfitting summary
        overfitting_detected = sum(1 for v in self.validation_history
                                 if v.overfitting_analysis and v.overfitting_analysis.get('is_overfitting', False))

        return {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'success_rate': success_rate,
            'model_type_distribution': model_type_counts,
            'average_validation_score': avg_validation_score,
            'total_critical_issues': total_critical_issues,
            'total_warnings': total_warnings,
            'overfitting_detected_count': overfitting_detected,
            'overfitting_rate': overfitting_detected / total_validations if total_validations > 0 else 0
        }

    def get_validation_trends(self) -> Dict[str, Any]:
        """Get validation trends over time."""
        if not self.validation_history:
            return {'message': 'No validation history available'}

        # Group by time periods
        validation_trends = {}
        for validation in self.validation_history:
            date_key = validation.validation_timestamp.date().isoformat()
            if date_key not in validation_trends:
                validation_trends[date_key] = {
                    'total': 0,
                    'passed': 0,
                    'scores': [],
                    'critical_issues': 0,
                    'warnings': 0
                }

            validation_trends[date_key]['total'] += 1
            if validation.validation_passed:
                validation_trends[date_key]['passed'] += 1
            validation_trends[date_key]['scores'].append(validation.validation_score)
            validation_trends[date_key]['critical_issues'] += len(validation.critical_issues)
            validation_trends[date_key]['warnings'] += len(validation.warnings)

        # Calculate trends
        for date_key in validation_trends:
            trend = validation_trends[date_key]
            trend['success_rate'] = trend['passed'] / trend['total'] if trend['total'] > 0 else 0
            trend['avg_score'] = np.mean(trend['scores']) if trend['scores'] else 0

        return validation_trends

# Global integrator instance
DEFAULT_VALIDATION_REPORTING_INTEGRATOR = ValidationReportingIntegrator()

def get_validation_reporting_integrator() -> ValidationReportingIntegrator:
    """Get validation reporting integrator instance."""
    return DEFAULT_VALIDATION_REPORTING_INTEGRATOR

def process_validation_with_reporting(validation_report: Union[UniversalMLValidationReport, Dict[str, Any]],
                                    model_name: str = "unknown",
                                    model_type: str = "unknown",
                                    fold_number: Optional[int] = None,
                                    trial_number: Optional[int] = None,
                                    validation_duration: Optional[float] = None,
                                    model_metadata: Optional[Dict[str, Any]] = None) -> ValidationReportData:
    """Convenience function to process validation with reporting."""
    integrator = get_validation_reporting_integrator()
    return integrator.process_validation_report(
        validation_report, model_name, model_type, fold_number,
        trial_number, validation_duration, model_metadata
    )
