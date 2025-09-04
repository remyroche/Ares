#!/usr/bin/env python3
"""
Pipeline Validation Framework

This module provides comprehensive validation for the model-training pipeline,
ensuring each step leads to the next with proper validators, decorators, and utilities.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import pandas as pd
import numpy as np

from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
    safe_file_exists,
    safe_json_load,
    safe_json_dump,
    ensure_directory,
)
from src.utils.logger import system_logger


class ValidationLevel(Enum):
    """Validation levels for different pipeline stages."""
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
class ValidationReport:
    """Comprehensive validation report."""
    step_name: str
    validation_level: ValidationLevel
    result: ValidationResult
    timestamp: str
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class BaseValidator(ABC):
    """Base class for all pipeline validators."""
    
    def __init__(self, name: str, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.name = name
        self.validation_level = validation_level
        self.logger = system_logger.getChild(f"Validator.{name}")
        
    @abstractmethod
    async def validate(self, data: Any, context: Dict[str, Any]) -> ValidationReport:
        """Validate data and return comprehensive report."""
        pass
    
    def _create_report(
        self,
        result: ValidationResult,
        duration: float,
        details: Dict[str, Any] = None,
        warnings: List[str] = None,
        errors: List[str] = None,
        recommendations: List[str] = None,
        metrics: Dict[str, Any] = None
    ) -> ValidationReport:
        """Create a standardized validation report."""
        return ValidationReport(
            step_name=self.name,
            validation_level=self.validation_level,
            result=result,
            timestamp=format_datetime(get_current_datetime()),
            duration=duration,
            details=details or {},
            warnings=warnings or [],
            errors=errors or [],
            recommendations=recommendations or [],
            metrics=metrics or {}
        )


class DataFormatValidator(BaseValidator):
    """Validator for data formatting operations."""
    
    def __init__(self):
        super().__init__("DataFormatValidator", ValidationLevel.CRITICAL)
    
    async def validate(self, data: Any, context: Dict[str, Any]) -> ValidationReport:
        """Validate data formatting."""
        start_time = time.time()
        details = {}
        warnings = []
        errors = []
        recommendations = []
        
        try:
            # Validate data type
            if not isinstance(data, (pd.DataFrame, np.ndarray, dict, list)):
                errors.append(f"Invalid data type: {type(data)}")
                result = ValidationResult.FAILED
            else:
                # Validate DataFrame structure
                if isinstance(data, pd.DataFrame):
                    if data.empty:
                        warnings.append("DataFrame is empty")
                    else:
                        details["shape"] = data.shape
                        details["columns"] = list(data.columns)
                        details["dtypes"] = data.dtypes.to_dict()
                        
                        # Check for missing values
                        missing_count = data.isnull().sum().sum()
                        if missing_count > 0:
                            warnings.append(f"Found {missing_count} missing values")
                            details["missing_values"] = data.isnull().sum().to_dict()
                        
                        # Check for duplicates
                        duplicate_count = data.duplicated().sum()
                        if duplicate_count > 0:
                            warnings.append(f"Found {duplicate_count} duplicate rows")
                            details["duplicate_count"] = duplicate_count
                
                # Validate data quality
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Check for infinite values
                    inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
                    if inf_count > 0:
                        errors.append(f"Found {inf_count} infinite values")
                        details["infinite_values"] = inf_count
                    
                    # Check for negative values where they shouldn't be
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if 'price' in col.lower() or 'volume' in col.lower():
                            negative_count = (data[col] < 0).sum()
                            if negative_count > 0:
                                warnings.append(f"Found {negative_count} negative values in {col}")
                
                result = ValidationResult.PASSED if not errors else ValidationResult.FAILED
                
                # Add recommendations
                if warnings:
                    recommendations.append("Review data quality issues")
                if errors:
                    recommendations.append("Fix critical data issues before proceeding")
                
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            result = ValidationResult.FAILED
            self.logger.exception(f"Data format validation failed: {e}")
        
        duration = time.time() - start_time
        return self._create_report(
            result=result,
            duration=duration,
            details=details,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations
        )


class DataAnalysisValidator(BaseValidator):
    """Validator for data analysis operations."""
    
    def __init__(self):
        super().__init__("DataAnalysisValidator", ValidationLevel.COMPREHENSIVE)
    
    async def validate(self, data: Any, context: Dict[str, Any]) -> ValidationReport:
        """Validate data analysis results."""
        start_time = time.time()
        details = {}
        warnings = []
        errors = []
        recommendations = []
        
        try:
            if isinstance(data, dict):
                # Validate analysis results structure
                required_keys = ['statistics', 'insights', 'recommendations']
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    errors.append(f"Missing required analysis keys: {missing_keys}")
                
                # Validate statistics
                if 'statistics' in data:
                    stats = data['statistics']
                    if not isinstance(stats, dict):
                        errors.append("Statistics must be a dictionary")
                    else:
                        details["statistics_keys"] = list(stats.keys())
                        
                        # Check for required statistical measures
                        required_stats = ['mean', 'std', 'min', 'max']
                        missing_stats = [stat for stat in required_stats if stat not in stats]
                        if missing_stats:
                            warnings.append(f"Missing statistical measures: {missing_stats}")
                
                # Validate insights
                if 'insights' in data:
                    insights = data['insights']
                    if not isinstance(insights, list):
                        errors.append("Insights must be a list")
                    else:
                        details["insight_count"] = len(insights)
                        if len(insights) == 0:
                            warnings.append("No insights generated")
                
                # Validate recommendations
                if 'recommendations' in data:
                    recs = data['recommendations']
                    if not isinstance(recs, list):
                        errors.append("Recommendations must be a list")
                    else:
                        details["recommendation_count"] = len(recs)
                        if len(recs) == 0:
                            warnings.append("No recommendations generated")
                
                result = ValidationResult.PASSED if not errors else ValidationResult.FAILED
                
            else:
                errors.append(f"Invalid analysis data type: {type(data)}")
                result = ValidationResult.FAILED
                
        except Exception as e:
            errors.append(f"Analysis validation error: {str(e)}")
            result = ValidationResult.FAILED
            self.logger.exception(f"Data analysis validation failed: {e}")
        
        duration = time.time() - start_time
        return self._create_report(
            result=result,
            duration=duration,
            details=details,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations
        )


class ModelTrainingValidator(BaseValidator):
    """Validator for model training operations."""
    
    def __init__(self):
        super().__init__("ModelTrainingValidator", ValidationLevel.CRITICAL)
    
    async def validate(self, data: Any, context: Dict[str, Any]) -> ValidationReport:
        """Validate model training results."""
        start_time = time.time()
        details = {}
        warnings = []
        errors = []
        recommendations = []
        
        try:
            if isinstance(data, dict):
                # Validate model training results
                required_keys = ['model', 'metrics', 'training_history']
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    errors.append(f"Missing required training keys: {missing_keys}")
                
                # Validate model
                if 'model' in data:
                    model = data['model']
                    if model is None:
                        errors.append("Model is None")
                    else:
                        details["model_type"] = type(model).__name__
                        details["model_attributes"] = dir(model)
                
                # Validate metrics
                if 'metrics' in data:
                    metrics = data['metrics']
                    if not isinstance(metrics, dict):
                        errors.append("Metrics must be a dictionary")
                    else:
                        details["metrics_keys"] = list(metrics.keys())
                        
                        # Check for required metrics
                        required_metrics = ['accuracy', 'loss']
                        missing_metrics = [metric for metric in required_metrics if metric not in metrics]
                        if missing_metrics:
                            warnings.append(f"Missing metrics: {missing_metrics}")
                        
                        # Validate metric values
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, (int, float)):
                                if np.isnan(metric_value) or np.isinf(metric_value):
                                    errors.append(f"Invalid metric value for {metric_name}: {metric_value}")
                                elif metric_name == 'accuracy' and (metric_value < 0 or metric_value > 1):
                                    warnings.append(f"Accuracy out of range [0,1]: {metric_value}")
                
                # Validate training history
                if 'training_history' in data:
                    history = data['training_history']
                    if not isinstance(history, dict):
                        errors.append("Training history must be a dictionary")
                    else:
                        details["history_keys"] = list(history.keys())
                        if len(history) == 0:
                            warnings.append("Empty training history")
                
                result = ValidationResult.PASSED if not errors else ValidationResult.FAILED
                
            else:
                errors.append(f"Invalid training data type: {type(data)}")
                result = ValidationResult.FAILED
                
        except Exception as e:
            errors.append(f"Model training validation error: {str(e)}")
            result = ValidationResult.FAILED
            self.logger.exception(f"Model training validation failed: {e}")
        
        duration = time.time() - start_time
        return self._create_report(
            result=result,
            duration=duration,
            details=details,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations
        )


class DataAccessValidator(BaseValidator):
    """Validator for data access operations."""
    
    def __init__(self):
        super().__init__("DataAccessValidator", ValidationLevel.CRITICAL)
    
    async def validate(self, data: Any, context: Dict[str, Any]) -> ValidationReport:
        """Validate data access operations."""
        start_time = time.time()
        details = {}
        warnings = []
        errors = []
        recommendations = []
        
        try:
            # Validate file paths
            if 'file_paths' in context:
                file_paths = context['file_paths']
                if not isinstance(file_paths, list):
                    errors.append("File paths must be a list")
                else:
                    details["file_count"] = len(file_paths)
                    missing_files = []
                    for file_path in file_paths:
                        if not safe_file_exists(file_path):
                            missing_files.append(file_path)
                    
                    if missing_files:
                        errors.append(f"Missing files: {missing_files}")
                        details["missing_files"] = missing_files
                    else:
                        details["all_files_exist"] = True
            
            # Validate data access permissions
            if 'data_dir' in context:
                data_dir = context['data_dir']
                if not safe_file_exists(data_dir):
                    errors.append(f"Data directory does not exist: {data_dir}")
                else:
                    details["data_dir"] = data_dir
                    details["data_dir_exists"] = True
            
            # Validate data format
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    warnings.append("DataFrame is empty")
                else:
                    details["data_shape"] = data.shape
                    details["data_columns"] = list(data.columns)
            
            result = ValidationResult.PASSED if not errors else ValidationResult.FAILED
            
            if warnings:
                recommendations.append("Review data access warnings")
            if errors:
                recommendations.append("Fix data access issues before proceeding")
                
        except Exception as e:
            errors.append(f"Data access validation error: {str(e)}")
            result = ValidationResult.FAILED
            self.logger.exception(f"Data access validation failed: {e}")
        
        duration = time.time() - start_time
        return self._create_report(
            result=result,
            duration=duration,
            details=details,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations
        )


class PipelineValidationOrchestrator:
    """Orchestrator for pipeline validation."""
    
    def __init__(self):
        self.logger = system_logger.getChild("PipelineValidationOrchestrator")
        self.validators = {
            'data_format': DataFormatValidator(),
            'data_analysis': DataAnalysisValidator(),
            'model_training': ModelTrainingValidator(),
            'data_access': DataAccessValidator(),
        }
        self.validation_reports = []
    
    async def validate_step(
        self,
        step_name: str,
        data: Any,
        context: Dict[str, Any],
        validation_types: List[str] = None
    ) -> Dict[str, ValidationReport]:
        """Validate a pipeline step with multiple validators."""
        if validation_types is None:
            validation_types = list(self.validators.keys())
        
        self.logger.info(f"🔍 Validating step: {step_name}")
        results = {}
        
        for validation_type in validation_types:
            if validation_type in self.validators:
                validator = self.validators[validation_type]
                try:
                    report = await validator.validate(data, context)
                    results[validation_type] = report
                    self.validation_reports.append(report)
                    
                    # Log validation result
                    if report.result == ValidationResult.PASSED:
                        self.logger.info(f"✅ {validation_type} validation passed for {step_name}")
                    elif report.result == ValidationResult.WARNING:
                        self.logger.warning(f"⚠️ {validation_type} validation warnings for {step_name}")
                    else:
                        self.logger.error(f"❌ {validation_type} validation failed for {step_name}")
                        
                except Exception as e:
                    self.logger.exception(f"❌ {validation_type} validation error for {step_name}: {e}")
                    error_report = ValidationReport(
                        step_name=step_name,
                        validation_level=ValidationLevel.CRITICAL,
                        result=ValidationResult.FAILED,
                        timestamp=format_datetime(get_current_datetime()),
                        duration=0.0,
                        errors=[f"Validation error: {str(e)}"]
                    )
                    results[validation_type] = error_report
                    self.validation_reports.append(error_report)
        
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation reports."""
        total_validations = len(self.validation_reports)
        passed = sum(1 for r in self.validation_reports if r.result == ValidationResult.PASSED)
        failed = sum(1 for r in self.validation_reports if r.result == ValidationResult.FAILED)
        warnings = sum(1 for r in self.validation_reports if r.result == ValidationResult.WARNING)
        
        return {
            'total_validations': total_validations,
            'passed': passed,
            'failed': failed,
            'warnings': warnings,
            'success_rate': passed / total_validations if total_validations > 0 else 0,
            'reports': self.validation_reports
        }
    
    def save_validation_report(self, file_path: str) -> None:
        """Save validation reports to file."""
        report_data = {
            'summary': self.get_validation_summary(),
            'timestamp': format_datetime(get_current_datetime()),
            'reports': [
                {
                    'step_name': r.step_name,
                    'validation_level': r.validation_level.value,
                    'result': r.result.value,
                    'timestamp': r.timestamp,
                    'duration': r.duration,
                    'details': r.details,
                    'warnings': r.warnings,
                    'errors': r.errors,
                    'recommendations': r.recommendations,
                    'metrics': r.metrics
                }
                for r in self.validation_reports
            ]
        }
        
        safe_json_dump(report_data, file_path, indent=2)
        self.logger.info(f"💾 Validation report saved to: {file_path}")


# Global validation orchestrator instance
validation_orchestrator = PipelineValidationOrchestrator()