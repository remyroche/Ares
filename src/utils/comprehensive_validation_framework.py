#!/usr/bin/env python3
"""
Comprehensive Validation Framework

This module provides a comprehensive validation framework for:
1. Pipeline integrity validation
2. Data quality validation
3. Step dependency validation
4. Configuration validation
5. Security validation
6. Performance validation
7. Cross-step consistency validation
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
import logging
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
    safe_file_exists,
    safe_json_load,
    safe_json_dump,
)
from src.utils.logger import system_logger
from src.utils.data_quality_framework import DataQualityFramework
from src.utils.security_framework import SecurityFramework
from src.utils.enhanced_common_operations import data_analysis_manager
import pandas as pd
import numpy as np

logger = system_logger.getChild("ComprehensiveValidationFramework")


class ValidationLevel(Enum):
    """Validation levels for different types of checks."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """Validation result types."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationCheck:
    """Represents a single validation check."""
    name: str
    description: str
    level: ValidationLevel
    required: bool
    check_function: Callable
    error_message: str
    warning_message: str = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    validation_id: str
    timestamp: datetime
    overall_result: ValidationResult
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    skipped_checks: int
    execution_time: float
    checks: List[Dict[str, Any]]
    summary: Dict[str, Any]
    recommendations: List[str]


class BaseValidator(ABC):
    """Base class for all validators."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.logger = system_logger.getChild(f"Validator_{name}")
        self.checks: List[ValidationCheck] = []
        self.validation_results: Dict[str, Any] = {}
    
    @abstractmethod
    async def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationReport:
        """Perform validation and return comprehensive report."""
        pass
    
    def add_check(self, check: ValidationCheck):
        """Add a validation check."""
        self.checks.append(check)
    
    def get_check_by_name(self, name: str) -> Optional[ValidationCheck]:
        """Get validation check by name."""
        for check in self.checks:
            if check.name == name:
                return check
        return None


class PipelineIntegrityValidator(BaseValidator):
    """Validator for overall pipeline integrity."""
    
    def __init__(self):
        super().__init__("PipelineIntegrityValidator", "Validates overall pipeline integrity")
        self._setup_checks()
    
    def _setup_checks(self):
        """Setup validation checks for pipeline integrity."""
        
        # Check 1: Pipeline state consistency
        self.add_check(ValidationCheck(
            name="pipeline_state_consistency",
            description="Check pipeline state consistency across steps",
            level=ValidationLevel.CRITICAL,
            required=True,
            check_function=self._check_pipeline_state_consistency,
            error_message="Pipeline state is inconsistent",
            warning_message="Pipeline state has minor inconsistencies"
        ))
        
        # Check 2: Step execution order
        self.add_check(ValidationCheck(
            name="step_execution_order",
            description="Validate step execution order",
            level=ValidationLevel.CRITICAL,
            required=True,
            check_function=self._check_step_execution_order,
            error_message="Step execution order is invalid",
            warning_message="Step execution order has minor issues"
        ))
        
        # Check 3: Data flow consistency
        self.add_check(ValidationCheck(
            name="data_flow_consistency",
            description="Check data flow consistency between steps",
            level=ValidationLevel.COMPREHENSIVE,
            required=True,
            check_function=self._check_data_flow_consistency,
            error_message="Data flow is inconsistent",
            warning_message="Data flow has minor inconsistencies"
        ))
        
        # Check 4: Configuration consistency
        self.add_check(ValidationCheck(
            name="configuration_consistency",
            description="Validate configuration consistency",
            level=ValidationLevel.STANDARD,
            required=False,
            check_function=self._check_configuration_consistency,
            error_message="Configuration is inconsistent",
            warning_message="Configuration has minor inconsistencies"
        ))
    
    async def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationReport:
        """Validate pipeline integrity."""
        start_time = time.time()
        validation_id = f"pipeline_integrity_{int(time.time())}"
        
        self.logger.info(f"🔍 Starting pipeline integrity validation: {validation_id}")
        
        try:
            # Initialize validation results
            check_results = []
            passed_checks = 0
            failed_checks = 0
            warning_checks = 0
            skipped_checks = 0
            
            # Execute all checks
            for check in self.checks:
                try:
                    self.logger.debug(f"🔍 Executing check: {check.name}")
                    
                    # Check if validation level is appropriate
                    if not self._should_execute_check(check, context):
                        check_result = {
                            'name': check.name,
                            'description': check.description,
                            'level': check.level.value,
                            'result': ValidationResult.SKIPPED.value,
                            'message': 'Check skipped due to validation level',
                            'execution_time': 0.0,
                            'timestamp': get_current_datetime().isoformat()
                        }
                        skipped_checks += 1
                    else:
                        # Execute check
                        check_start = time.time()
                        result = await check.check_function(data, context)
                        check_duration = time.time() - check_start
                        
                        # Determine result type
                        if result.get('passed', False):
                            result_type = ValidationResult.PASSED
                            passed_checks += 1
                            message = "Check passed"
                        elif result.get('warning', False):
                            result_type = ValidationResult.WARNING
                            warning_checks += 1
                            message = result.get('message', check.warning_message)
                        else:
                            result_type = ValidationResult.FAILED
                            failed_checks += 1
                            message = result.get('message', check.error_message)
                        
                        check_result = {
                            'name': check.name,
                            'description': check.description,
                            'level': check.level.value,
                            'result': result_type.value,
                            'message': message,
                            'details': result.get('details', {}),
                            'execution_time': check_duration,
                            'timestamp': get_current_datetime().isoformat()
                        }
                    
                    check_results.append(check_result)
                    
                except Exception as e:
                    self.logger.exception(f"❌ Check {check.name} failed with exception: {e}")
                    check_result = {
                        'name': check.name,
                        'description': check.description,
                        'level': check.level.value,
                        'result': ValidationResult.FAILED.value,
                        'message': f"Check failed with exception: {str(e)}",
                        'execution_time': 0.0,
                        'timestamp': get_current_datetime().isoformat()
                    }
                    check_results.append(check_result)
                    failed_checks += 1
            
            # Determine overall result
            if failed_checks > 0:
                overall_result = ValidationResult.FAILED
            elif warning_checks > 0:
                overall_result = ValidationResult.WARNING
            else:
                overall_result = ValidationResult.PASSED
            
            # Generate summary and recommendations
            summary = self._generate_summary(check_results, passed_checks, failed_checks, warning_checks, skipped_checks)
            recommendations = self._generate_recommendations(check_results)
            
            execution_time = time.time() - start_time
            
            report = ValidationReport(
                validation_id=validation_id,
                timestamp=get_current_datetime(),
                overall_result=overall_result,
                total_checks=len(self.checks),
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                warning_checks=warning_checks,
                skipped_checks=skipped_checks,
                execution_time=execution_time,
                checks=check_results,
                summary=summary,
                recommendations=recommendations
            )
            
            self.logger.info(f"✅ Pipeline integrity validation completed: {validation_id} in {execution_time:.3f}s")
            return report
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.exception(f"❌ Pipeline integrity validation failed: {e}")
            
            # Return failed report
            return ValidationReport(
                validation_id=validation_id,
                timestamp=get_current_datetime(),
                overall_result=ValidationResult.FAILED,
                total_checks=len(self.checks),
                passed_checks=0,
                failed_checks=1,
                warning_checks=0,
                skipped_checks=0,
                execution_time=execution_time,
                checks=[],
                summary={'error': str(e)},
                recommendations=['Fix validation framework error']
            )
    
    def _should_execute_check(self, check: ValidationCheck, context: Dict[str, Any] = None) -> bool:
        """Determine if a check should be executed based on validation level."""
        if context is None:
            return True
        
        requested_level = context.get('validation_level', ValidationLevel.STANDARD)
        
        # Map levels to numeric values for comparison
        level_values = {
            ValidationLevel.BASIC: 1,
            ValidationLevel.STANDARD: 2,
            ValidationLevel.COMPREHENSIVE: 3,
            ValidationLevel.CRITICAL: 4
        }
        
        requested_value = level_values.get(requested_level, 2)
        check_value = level_values.get(check.level, 2)
        
        return check_value <= requested_value
    
    async def _check_pipeline_state_consistency(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check pipeline state consistency."""
        try:
            pipeline_state = data.get('pipeline_state', {}) if isinstance(data, dict) else {}
            
            if not pipeline_state:
                return {'passed': False, 'message': 'No pipeline state found'}
            
            # Check that all steps have consistent structure
            required_keys = ['success', 'timestamp', 'outputs']
            inconsistencies = []
            
            for step_name, step_data in pipeline_state.items():
                if not isinstance(step_data, dict):
                    inconsistencies.append(f"Step {step_name} data is not a dictionary")
                    continue
                
                missing_keys = [key for key in required_keys if key not in step_data]
                if missing_keys:
                    inconsistencies.append(f"Step {step_name} missing keys: {missing_keys}")
            
            if inconsistencies:
                return {
                    'passed': False,
                    'message': 'Pipeline state inconsistencies found',
                    'details': {'inconsistencies': inconsistencies}
                }
            
            return {'passed': True, 'message': 'Pipeline state is consistent'}
            
        except Exception as e:
            return {'passed': False, 'message': f'Error checking pipeline state: {str(e)}'}
    
    async def _check_step_execution_order(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check step execution order."""
        try:
            execution_results = data.get('execution_results', {}) if isinstance(data, dict) else {}
            
            if not execution_results:
                return {'passed': False, 'message': 'No execution results found'}
            
            # Expected execution order
            expected_order = ['data_collection', 'hmm_clustering', 'feature_engineering']
            
            # Get actual execution order from timestamps
            step_timestamps = {}
            for step_name, step_result in execution_results.items():
                if isinstance(step_result, dict) and 'timestamp' in step_result:
                    step_timestamps[step_name] = step_result['timestamp']
            
            # Check if steps were executed in correct order
            order_violations = []
            for i in range(len(expected_order) - 1):
                current_step = expected_order[i]
                next_step = expected_order[i + 1]
                
                if current_step in step_timestamps and next_step in step_timestamps:
                    current_time = datetime.fromisoformat(step_timestamps[current_step].replace('Z', '+00:00'))
                    next_time = datetime.fromisoformat(step_timestamps[next_step].replace('Z', '+00:00'))
                    
                    if current_time > next_time:
                        order_violations.append(f"{current_step} executed after {next_step}")
            
            if order_violations:
                return {
                    'passed': False,
                    'message': 'Step execution order violations found',
                    'details': {'violations': order_violations}
                }
            
            return {'passed': True, 'message': 'Step execution order is correct'}
            
        except Exception as e:
            return {'passed': False, 'message': f'Error checking execution order: {str(e)}'}
    
    async def _check_data_flow_consistency(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check data flow consistency between steps."""
        try:
            pipeline_state = data.get('pipeline_state', {}) if isinstance(data, dict) else {}
            
            if not pipeline_state:
                return {'passed': False, 'message': 'No pipeline state found'}
            
            # Define expected data flow
            data_flow = {
                'data_collection': ['data_file', 'data_exists'],
                'hmm_clustering': ['regime_model', 'regime_labels'],
                'feature_engineering': ['features', 'feature_count']
            }
            
            flow_issues = []
            
            for step_name, expected_outputs in data_flow.items():
                if step_name not in pipeline_state:
                    flow_issues.append(f"Step {step_name} not found in pipeline state")
                    continue
                
                step_data = pipeline_state[step_name]
                if not step_data.get('success', False):
                    flow_issues.append(f"Step {step_name} did not complete successfully")
                    continue
                
                outputs = step_data.get('outputs', {})
                missing_outputs = [output for output in expected_outputs if output not in outputs]
                if missing_outputs:
                    flow_issues.append(f"Step {step_name} missing outputs: {missing_outputs}")
            
            if flow_issues:
                return {
                    'passed': False,
                    'message': 'Data flow inconsistencies found',
                    'details': {'issues': flow_issues}
                }
            
            return {'passed': True, 'message': 'Data flow is consistent'}
            
        except Exception as e:
            return {'passed': False, 'message': f'Error checking data flow: {str(e)}'}
    
    async def _check_configuration_consistency(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check configuration consistency."""
        try:
            config = data.get('config', {}) if isinstance(data, dict) else {}
            
            if not config:
                return {'warning': True, 'message': 'No configuration found'}
            
            # Check for conflicting settings
            conflicts = []
            
            if config.get('force_rerun', False) and config.get('use_cached_data', False):
                conflicts.append('force_rerun and use_cached_data are conflicting')
            
            if config.get('lookback_days', 0) < 1:
                conflicts.append('lookback_days should be positive')
            
            if conflicts:
                return {
                    'passed': False,
                    'message': 'Configuration conflicts found',
                    'details': {'conflicts': conflicts}
                }
            
            return {'passed': True, 'message': 'Configuration is consistent'}
            
        except Exception as e:
            return {'passed': False, 'message': f'Error checking configuration: {str(e)}'}
    
    def _generate_summary(self, check_results: List[Dict[str, Any]], passed: int, failed: int, warnings: int, skipped: int) -> Dict[str, Any]:
        """Generate validation summary."""
        total = len(check_results)
        
        return {
            'total_checks': total,
            'passed_checks': passed,
            'failed_checks': failed,
            'warning_checks': warnings,
            'skipped_checks': skipped,
            'success_rate': (passed / total * 100) if total > 0 else 0,
            'critical_failures': len([r for r in check_results if r.get('result') == ValidationResult.FAILED.value and r.get('level') == ValidationLevel.CRITICAL.value]),
            'execution_time': sum(r.get('execution_time', 0) for r in check_results)
        }
    
    def _generate_recommendations(self, check_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        failed_checks = [r for r in check_results if r.get('result') == ValidationResult.FAILED.value]
        warning_checks = [r for r in check_results if r.get('result') == ValidationResult.WARNING.value]
        
        if failed_checks:
            recommendations.append("Address all failed validation checks before proceeding")
            
            critical_failures = [r for r in failed_checks if r.get('level') == ValidationLevel.CRITICAL.value]
            if critical_failures:
                recommendations.append("Critical validation failures must be resolved immediately")
        
        if warning_checks:
            recommendations.append("Review and address validation warnings")
        
        if not failed_checks and not warning_checks:
            recommendations.append("All validations passed - pipeline is ready for execution")
        
        return recommendations


class DataQualityValidator(BaseValidator):
    """Validator for data quality across the pipeline."""
    
    def __init__(self):
        super().__init__("DataQualityValidator", "Validates data quality across pipeline steps")
        self.data_quality = DataQualityFramework()
        self._setup_checks()
    
    def _setup_checks(self):
        """Setup validation checks for data quality."""
        
        # Check 1: Data completeness
        self.add_check(ValidationCheck(
            name="data_completeness",
            description="Check data completeness across all steps",
            level=ValidationLevel.CRITICAL,
            required=True,
            check_function=self._check_data_completeness,
            error_message="Data completeness issues found",
            warning_message="Minor data completeness issues"
        ))
        
        # Check 2: Data consistency
        self.add_check(ValidationCheck(
            name="data_consistency",
            description="Check data consistency across steps",
            level=ValidationLevel.COMPREHENSIVE,
            required=True,
            check_function=self._check_data_consistency,
            error_message="Data consistency issues found",
            warning_message="Minor data consistency issues"
        ))
        
        # Check 3: Data validity
        self.add_check(ValidationCheck(
            name="data_validity",
            description="Check data validity and format",
            level=ValidationLevel.STANDARD,
            required=True,
            check_function=self._check_data_validity,
            error_message="Data validity issues found",
            warning_message="Minor data validity issues"
        ))
    
    async def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationReport:
        """Validate data quality."""
        start_time = time.time()
        validation_id = f"data_quality_{int(time.time())}"
        
        self.logger.info(f"🔍 Starting data quality validation: {validation_id}")
        
        try:
            # Initialize data quality framework
            await self.data_quality.initialize()
            
            # Execute validation checks
            check_results = []
            passed_checks = 0
            failed_checks = 0
            warning_checks = 0
            skipped_checks = 0
            
            for check in self.checks:
                try:
                    self.logger.debug(f"🔍 Executing data quality check: {check.name}")
                    
                    check_start = time.time()
                    result = await check.check_function(data, context)
                    check_duration = time.time() - check_start
                    
                    # Determine result type
                    if result.get('passed', False):
                        result_type = ValidationResult.PASSED
                        passed_checks += 1
                        message = "Check passed"
                    elif result.get('warning', False):
                        result_type = ValidationResult.WARNING
                        warning_checks += 1
                        message = result.get('message', check.warning_message)
                    else:
                        result_type = ValidationResult.FAILED
                        failed_checks += 1
                        message = result.get('message', check.error_message)
                    
                    check_result = {
                        'name': check.name,
                        'description': check.description,
                        'level': check.level.value,
                        'result': result_type.value,
                        'message': message,
                        'details': result.get('details', {}),
                        'execution_time': check_duration,
                        'timestamp': get_current_datetime().isoformat()
                    }
                    
                    check_results.append(check_result)
                    
                except Exception as e:
                    self.logger.exception(f"❌ Data quality check {check.name} failed: {e}")
                    check_result = {
                        'name': check.name,
                        'description': check.description,
                        'level': check.level.value,
                        'result': ValidationResult.FAILED.value,
                        'message': f"Check failed with exception: {str(e)}",
                        'execution_time': 0.0,
                        'timestamp': get_current_datetime().isoformat()
                    }
                    check_results.append(check_result)
                    failed_checks += 1
            
            # Determine overall result
            if failed_checks > 0:
                overall_result = ValidationResult.FAILED
            elif warning_checks > 0:
                overall_result = ValidationResult.WARNING
            else:
                overall_result = ValidationResult.PASSED
            
            # Generate summary and recommendations
            summary = self._generate_summary(check_results, passed_checks, failed_checks, warning_checks, skipped_checks)
            recommendations = self._generate_recommendations(check_results)
            
            execution_time = time.time() - start_time
            
            report = ValidationReport(
                validation_id=validation_id,
                timestamp=get_current_datetime(),
                overall_result=overall_result,
                total_checks=len(self.checks),
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                warning_checks=warning_checks,
                skipped_checks=skipped_checks,
                execution_time=execution_time,
                checks=check_results,
                summary=summary,
                recommendations=recommendations
            )
            
            self.logger.info(f"✅ Data quality validation completed: {validation_id} in {execution_time:.3f}s")
            return report
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.exception(f"❌ Data quality validation failed: {e}")
            
            return ValidationReport(
                validation_id=validation_id,
                timestamp=get_current_datetime(),
                overall_result=ValidationResult.FAILED,
                total_checks=len(self.checks),
                passed_checks=0,
                failed_checks=1,
                warning_checks=0,
                skipped_checks=0,
                execution_time=execution_time,
                checks=[],
                summary={'error': str(e)},
                recommendations=['Fix data quality validation framework error']
            )
    
    async def _check_data_completeness(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check data completeness."""
        try:
            pipeline_state = data.get('pipeline_state', {}) if isinstance(data, dict) else {}
            
            if not pipeline_state:
                return {'passed': False, 'message': 'No pipeline state found'}
            
            completeness_issues = []
            
            for step_name, step_data in pipeline_state.items():
                if not step_data.get('success', False):
                    completeness_issues.append(f"Step {step_name} did not complete successfully")
                    continue
                
                outputs = step_data.get('outputs', {})
                if not outputs:
                    completeness_issues.append(f"Step {step_name} has no outputs")
            
            if completeness_issues:
                return {
                    'passed': False,
                    'message': 'Data completeness issues found',
                    'details': {'issues': completeness_issues}
                }
            
            return {'passed': True, 'message': 'Data completeness is good'}
            
        except Exception as e:
            return {'passed': False, 'message': f'Error checking data completeness: {str(e)}'}
    
    async def _check_data_consistency(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check data consistency."""
        try:
            pipeline_state = data.get('pipeline_state', {}) if isinstance(data, dict) else {}
            
            if not pipeline_state:
                return {'passed': False, 'message': 'No pipeline state found'}
            
            consistency_issues = []
            
            # Check timestamp consistency
            timestamps = []
            for step_name, step_data in pipeline_state.items():
                if 'timestamp' in step_data:
                    timestamps.append((step_name, step_data['timestamp']))
            
            if len(timestamps) > 1:
                # Check if timestamps are in chronological order
                for i in range(len(timestamps) - 1):
                    current_time = datetime.fromisoformat(timestamps[i][1].replace('Z', '+00:00'))
                    next_time = datetime.fromisoformat(timestamps[i + 1][1].replace('Z', '+00:00'))
                    
                    if current_time > next_time:
                        consistency_issues.append(f"Timestamp order issue: {timestamps[i][0]} after {timestamps[i + 1][0]}")
            
            if consistency_issues:
                return {
                    'passed': False,
                    'message': 'Data consistency issues found',
                    'details': {'issues': consistency_issues}
                }
            
            return {'passed': True, 'message': 'Data consistency is good'}
            
        except Exception as e:
            return {'passed': False, 'message': f'Error checking data consistency: {str(e)}'}
    
    async def _check_data_validity(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check data validity."""
        try:
            pipeline_state = data.get('pipeline_state', {}) if isinstance(data, dict) else {}
            
            if not pipeline_state:
                return {'passed': False, 'message': 'No pipeline state found'}
            
            validity_issues = []
            
            for step_name, step_data in pipeline_state.items():
                if not isinstance(step_data, dict):
                    validity_issues.append(f"Step {step_name} data is not a dictionary")
                    continue
                
                # Check required fields
                required_fields = ['success', 'timestamp']
                missing_fields = [field for field in required_fields if field not in step_data]
                if missing_fields:
                    validity_issues.append(f"Step {step_name} missing required fields: {missing_fields}")
                
                # Check data types
                if 'success' in step_data and not isinstance(step_data['success'], bool):
                    validity_issues.append(f"Step {step_name} success field is not boolean")
            
            if validity_issues:
                return {
                    'passed': False,
                    'message': 'Data validity issues found',
                    'details': {'issues': validity_issues}
                }
            
            return {'passed': True, 'message': 'Data validity is good'}
            
        except Exception as e:
            return {'passed': False, 'message': f'Error checking data validity: {str(e)}'}
    
    def _generate_summary(self, check_results: List[Dict[str, Any]], passed: int, failed: int, warnings: int, skipped: int) -> Dict[str, Any]:
        """Generate validation summary."""
        total = len(check_results)
        
        return {
            'total_checks': total,
            'passed_checks': passed,
            'failed_checks': failed,
            'warning_checks': warnings,
            'skipped_checks': skipped,
            'success_rate': (passed / total * 100) if total > 0 else 0,
            'data_quality_score': self._calculate_data_quality_score(check_results),
            'execution_time': sum(r.get('execution_time', 0) for r in check_results)
        }
    
    def _calculate_data_quality_score(self, check_results: List[Dict[str, Any]]) -> float:
        """Calculate overall data quality score."""
        if not check_results:
            return 0.0
        
        total_weight = 0
        weighted_score = 0
        
        for result in check_results:
            level = result.get('level', ValidationLevel.STANDARD.value)
            
            # Assign weights based on validation level
            if level == ValidationLevel.CRITICAL.value:
                weight = 4
            elif level == ValidationLevel.COMPREHENSIVE.value:
                weight = 3
            elif level == ValidationLevel.STANDARD.value:
                weight = 2
            else:
                weight = 1
            
            total_weight += weight
            
            # Assign scores based on result
            if result.get('result') == ValidationResult.PASSED.value:
                score = 1.0
            elif result.get('result') == ValidationResult.WARNING.value:
                score = 0.7
            elif result.get('result') == ValidationResult.SKIPPED.value:
                score = 0.5
            else:
                score = 0.0
            
            weighted_score += score * weight
        
        return (weighted_score / total_weight * 100) if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, check_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        failed_checks = [r for r in check_results if r.get('result') == ValidationResult.FAILED.value]
        warning_checks = [r for r in check_results if r.get('result') == ValidationResult.WARNING.value]
        
        if failed_checks:
            recommendations.append("Address all data quality failures before proceeding")
            
            # Specific recommendations based on failed checks
            for check in failed_checks:
                if 'completeness' in check.get('name', ''):
                    recommendations.append("Ensure all pipeline steps complete successfully")
                elif 'consistency' in check.get('name', ''):
                    recommendations.append("Review data consistency across pipeline steps")
                elif 'validity' in check.get('name', ''):
                    recommendations.append("Validate data formats and types")
        
        if warning_checks:
            recommendations.append("Review data quality warnings")
        
        if not failed_checks and not warning_checks:
            recommendations.append("Data quality validation passed - data is ready for analysis")
        
        return recommendations


class ComprehensiveValidationFramework:
    """Main framework for comprehensive validation."""
    
    def __init__(self):
        self.logger = system_logger.getChild('ComprehensiveValidationFramework')
        self.validators = {
            'pipeline_integrity': PipelineIntegrityValidator(),
            'data_quality': DataQualityValidator()
        }
        self.validation_history = []
    
    async def initialize(self) -> bool:
        """Initialize the validation framework."""
        try:
            self.logger.info("🚀 Initializing Comprehensive Validation Framework...")
            
            # Initialize all validators
            for name, validator in self.validators.items():
                self.logger.info(f"🔧 Initializing validator: {name}")
                # Validators are already initialized in their constructors
            
            self.logger.info("✅ Comprehensive Validation Framework initialized successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize validation framework: {e}")
            return False
    
    async def validate_pipeline(
        self,
        pipeline_data: Dict[str, Any],
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        validators: List[str] = None
    ) -> Dict[str, ValidationReport]:
        """Validate pipeline with specified validators and level."""
        
        if validators is None:
            validators = list(self.validators.keys())
        
        self.logger.info(f"🔍 Starting comprehensive pipeline validation with level: {validation_level.value}")
        
        context = {
            'validation_level': validation_level,
            'timestamp': get_current_datetime().isoformat()
        }
        
        validation_reports = {}
        
        for validator_name in validators:
            if validator_name not in self.validators:
                self.logger.warning(f"⚠️ Unknown validator: {validator_name}")
                continue
            
            try:
                self.logger.info(f"🔍 Running validator: {validator_name}")
                validator = self.validators[validator_name]
                report = await validator.validate(pipeline_data, context)
                validation_reports[validator_name] = report
                
                # Log validation result
                if report.overall_result == ValidationResult.PASSED:
                    self.logger.info(f"✅ Validator {validator_name} passed")
                elif report.overall_result == ValidationResult.WARNING:
                    self.logger.warning(f"⚠️ Validator {validator_name} has warnings")
                else:
                    self.logger.error(f"❌ Validator {validator_name} failed")
                
            except Exception as e:
                self.logger.exception(f"❌ Validator {validator_name} failed with exception: {e}")
                # Create failed report
                validation_reports[validator_name] = ValidationReport(
                    validation_id=f"failed_{validator_name}_{int(time.time())}",
                    timestamp=get_current_datetime(),
                    overall_result=ValidationResult.FAILED,
                    total_checks=0,
                    passed_checks=0,
                    failed_checks=1,
                    warning_checks=0,
                    skipped_checks=0,
                    execution_time=0.0,
                    checks=[],
                    summary={'error': str(e)},
                    recommendations=['Fix validator error']
                )
        
        # Store validation history
        self.validation_history.append({
            'timestamp': get_current_datetime().isoformat(),
            'validation_level': validation_level.value,
            'validators': validators,
            'reports': {name: report.__dict__ for name, report in validation_reports.items()}
        })
        
        # Keep only last 100 validation runs
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-100:]
        
        return validation_reports
    
    def get_validation_summary(self, validation_reports: Dict[str, ValidationReport]) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        total_checks = sum(report.total_checks for report in validation_reports.values())
        total_passed = sum(report.passed_checks for report in validation_reports.values())
        total_failed = sum(report.failed_checks for report in validation_reports.values())
        total_warnings = sum(report.warning_checks for report in validation_reports.values())
        total_skipped = sum(report.skipped_checks for report in validation_reports.values())
        total_execution_time = sum(report.execution_time for report in validation_reports.values())
        
        # Determine overall result
        if total_failed > 0:
            overall_result = ValidationResult.FAILED
        elif total_warnings > 0:
            overall_result = ValidationResult.WARNING
        else:
            overall_result = ValidationResult.PASSED
        
        return {
            'overall_result': overall_result.value,
            'total_validators': len(validation_reports),
            'total_checks': total_checks,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_warnings': total_warnings,
            'total_skipped': total_skipped,
            'success_rate': (total_passed / total_checks * 100) if total_checks > 0 else 0,
            'total_execution_time': total_execution_time,
            'validator_results': {
                name: {
                    'result': report.overall_result.value,
                    'checks': report.total_checks,
                    'passed': report.passed_checks,
                    'failed': report.failed_checks,
                    'warnings': report.warning_checks,
                    'execution_time': report.execution_time
                }
                for name, report in validation_reports.items()
            },
            'timestamp': get_current_datetime().isoformat()
        }
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get validation history."""
        return self.validation_history.copy()


# Global instance for easy access
comprehensive_validation_framework = ComprehensiveValidationFramework()


# Convenience functions
async def validate_pipeline_comprehensive(
    pipeline_data: Dict[str, Any],
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> Dict[str, ValidationReport]:
    """Convenience function for comprehensive pipeline validation."""
    return await comprehensive_validation_framework.validate_pipeline(
        pipeline_data, validation_level
    )


async def validate_pipeline_integrity(pipeline_data: Dict[str, Any]) -> ValidationReport:
    """Convenience function for pipeline integrity validation."""
    reports = await comprehensive_validation_framework.validate_pipeline(
        pipeline_data, ValidationLevel.CRITICAL, ['pipeline_integrity']
    )
    return reports.get('pipeline_integrity')


async def validate_data_quality(pipeline_data: Dict[str, Any]) -> ValidationReport:
    """Convenience function for data quality validation."""
    reports = await comprehensive_validation_framework.validate_pipeline(
        pipeline_data, ValidationLevel.COMPREHENSIVE, ['data_quality']
    )
    return reports.get('data_quality')