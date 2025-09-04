#!/usr/bin/env python3
"""
Comprehensive Pipeline Validator Framework

This module provides a robust validation framework for the Ares trading pipeline,
ensuring data integrity, step dependencies, and proper execution flow at each stage.
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

from src.core.decorators.errors import handles_errors, error_boundary
from src.core.decorators.validate import validates
from src.core.decorators.logging import logs_execution
from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    safe_file_exists,
    safe_json_load,
    safe_json_dump,
    ensure_directory
)


class ValidationLevel(Enum):
    """Validation levels for different pipeline stages."""
    CRITICAL = "critical"      # Must pass for pipeline to continue
    WARNING = "warning"        # Can continue with warnings
    INFO = "info"             # Informational only
    DEBUG = "debug"           # Debug information


class ValidationResult(Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationReport:
    """Comprehensive validation report for pipeline steps."""
    step_name: str
    validation_level: ValidationLevel
    result: ValidationResult
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: format_datetime(get_current_datetime()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation report to dictionary."""
        return {
            "step_name": self.step_name,
            "validation_level": self.validation_level.value,
            "result": self.result.value,
            "message": self.message,
            "details": self.details,
            "warnings": self.warnings,
            "errors": self.errors,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp
        }


class BaseValidator(ABC):
    """Base class for all pipeline validators."""
    
    def __init__(self, name: str, validation_level: ValidationLevel = ValidationLevel.CRITICAL):
        self.name = name
        self.validation_level = validation_level
        self.logger = logging.getLogger(f"validator.{name}")
    
    @abstractmethod
    async def validate(self, data: Any, context: Dict[str, Any]) -> ValidationReport:
        """Validate data and return validation report."""
        pass
    
    def _create_report(
        self,
        result: ValidationResult,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        errors: Optional[List[str]] = None,
        execution_time: float = 0.0
    ) -> ValidationReport:
        """Create a validation report."""
        return ValidationReport(
            step_name=self.name,
            validation_level=self.validation_level,
            result=result,
            message=message,
            details=details or {},
            warnings=warnings or [],
            errors=errors or [],
            execution_time=execution_time
        )


class DataFormatValidator(BaseValidator):
    """Validates data format and structure."""
    
    def __init__(self):
        super().__init__("data_format", ValidationLevel.CRITICAL)
    
    @handles_errors(ValueError, TypeError, fallback=None)
    @validates(strict=True)
    async def validate(self, data: Union[pd.DataFrame, Dict, List], context: Dict[str, Any]) -> ValidationReport:
        """Validate data format and structure."""
        start_time = time.time()
        
        try:
            if isinstance(data, pd.DataFrame):
                return await self._validate_dataframe(data, context)
            elif isinstance(data, dict):
                return await self._validate_dict(data, context)
            elif isinstance(data, list):
                return await self._validate_list(data, context)
            else:
                return self._create_report(
                    ValidationResult.FAILED,
                    f"Unsupported data type: {type(data)}",
                    errors=[f"Expected DataFrame, dict, or list, got {type(data)}"]
                )
        
        except Exception as e:
            return self._create_report(
                ValidationResult.FAILED,
                f"Data format validation failed: {str(e)}",
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    async def _validate_dataframe(self, df: pd.DataFrame, context: Dict[str, Any]) -> ValidationReport:
        """Validate DataFrame structure."""
        warnings = []
        errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
        
        # Check for required columns
        required_columns = context.get("required_columns", [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for duplicate columns
        if len(df.columns) != len(set(df.columns)):
            errors.append("Duplicate columns found")
        
        # Check for excessive null values
        null_threshold = context.get("null_threshold", 0.5)
        for col in df.columns:
            null_ratio = df[col].isnull().sum() / len(df)
            if null_ratio > null_threshold:
                warnings.append(f"Column '{col}' has {null_ratio:.2%} null values")
        
        # Check data types
        expected_types = context.get("expected_types", {})
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not self._is_compatible_type(actual_type, expected_type):
                    warnings.append(f"Column '{col}' type mismatch: expected {expected_type}, got {actual_type}")
        
        result = ValidationResult.PASSED if not errors else ValidationResult.FAILED
        if warnings and not errors:
            result = ValidationResult.WARNING
        
        return self._create_report(
            result,
            f"DataFrame validation {'passed' if result == ValidationResult.PASSED else 'failed'}",
            details={
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "null_counts": df.isnull().sum().to_dict()
            },
            warnings=warnings,
            errors=errors
        )
    
    async def _validate_dict(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationReport:
        """Validate dictionary structure."""
        warnings = []
        errors = []
        
        # Check required keys
        required_keys = context.get("required_keys", [])
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            errors.append(f"Missing required keys: {missing_keys}")
        
        # Check data types for specific keys
        expected_types = context.get("expected_types", {})
        for key, expected_type in expected_types.items():
            if key in data:
                if not isinstance(data[key], expected_type):
                    errors.append(f"Key '{key}' type mismatch: expected {expected_type}, got {type(data[key])}")
        
        result = ValidationResult.PASSED if not errors else ValidationResult.FAILED
        if warnings and not errors:
            result = ValidationResult.WARNING
        
        return self._create_report(
            result,
            f"Dictionary validation {'passed' if result == ValidationResult.PASSED else 'failed'}",
            details={"keys": list(data.keys()), "size": len(data)},
            warnings=warnings,
            errors=errors
        )
    
    async def _validate_list(self, data: List[Any], context: Dict[str, Any]) -> ValidationReport:
        """Validate list structure."""
        warnings = []
        errors = []
        
        # Check minimum length
        min_length = context.get("min_length", 0)
        if len(data) < min_length:
            errors.append(f"List too short: {len(data)} < {min_length}")
        
        # Check maximum length
        max_length = context.get("max_length")
        if max_length and len(data) > max_length:
            warnings.append(f"List longer than expected: {len(data)} > {max_length}")
        
        # Check element types
        expected_element_type = context.get("expected_element_type")
        if expected_element_type:
            for i, element in enumerate(data):
                if not isinstance(element, expected_element_type):
                    errors.append(f"Element {i} type mismatch: expected {expected_element_type}, got {type(element)}")
        
        result = ValidationResult.PASSED if not errors else ValidationResult.FAILED
        if warnings and not errors:
            result = ValidationResult.WARNING
        
        return self._create_report(
            result,
            f"List validation {'passed' if result == ValidationResult.PASSED else 'failed'}",
            details={"length": len(data)},
            warnings=warnings,
            errors=errors
        )
    
    def _is_compatible_type(self, actual: str, expected: str) -> bool:
        """Check if actual type is compatible with expected type."""
        type_mapping = {
            "int64": ["int", "integer", "int64"],
            "float64": ["float", "float64", "double"],
            "object": ["str", "string", "object"],
            "bool": ["bool", "boolean"],
            "datetime64[ns]": ["datetime", "datetime64", "timestamp"]
        }
        
        for compatible_types in type_mapping.values():
            if actual in compatible_types and expected in compatible_types:
                return True
        
        return actual == expected


class DataQualityValidator(BaseValidator):
    """Validates data quality metrics."""
    
    def __init__(self):
        super().__init__("data_quality", ValidationLevel.CRITICAL)
    
    @handles_errors(ValueError, TypeError, fallback=None)
    async def validate(self, data: pd.DataFrame, context: Dict[str, Any]) -> ValidationReport:
        """Validate data quality metrics."""
        start_time = time.time()
        
        try:
            warnings = []
            errors = []
            quality_metrics = {}
            
            # Check for missing values
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            quality_metrics["missing_ratio"] = missing_ratio
            
            if missing_ratio > 0.1:  # More than 10% missing
                warnings.append(f"High missing value ratio: {missing_ratio:.2%}")
            
            # Check for duplicate rows
            duplicate_ratio = data.duplicated().sum() / len(data)
            quality_metrics["duplicate_ratio"] = duplicate_ratio
            
            if duplicate_ratio > 0.05:  # More than 5% duplicates
                warnings.append(f"High duplicate ratio: {duplicate_ratio:.2%}")
            
            # Check for outliers in numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            outlier_counts = {}
            
            for col in numeric_columns:
                if len(data[col].dropna()) > 0:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                    outlier_ratio = outliers / len(data[col].dropna())
                    outlier_counts[col] = outlier_ratio
                    
                    if outlier_ratio > 0.1:  # More than 10% outliers
                        warnings.append(f"High outlier ratio in '{col}': {outlier_ratio:.2%}")
            
            quality_metrics["outlier_ratios"] = outlier_counts
            
            # Check data consistency
            consistency_issues = await self._check_consistency(data, context)
            if consistency_issues:
                warnings.extend(consistency_issues)
            
            result = ValidationResult.PASSED if not errors else ValidationResult.FAILED
            if warnings and not errors:
                result = ValidationResult.WARNING
            
            return self._create_report(
                result,
                f"Data quality validation {'passed' if result == ValidationResult.PASSED else 'failed'}",
                details=quality_metrics,
                warnings=warnings,
                errors=errors,
                execution_time=time.time() - start_time
            )
        
        except Exception as e:
            return self._create_report(
                ValidationResult.FAILED,
                f"Data quality validation failed: {str(e)}",
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    async def _check_consistency(self, data: pd.DataFrame, context: Dict[str, Any]) -> List[str]:
        """Check data consistency rules."""
        issues = []
        
        # Check for negative prices (if price columns exist)
        price_columns = [col for col in data.columns if 'price' in col.lower() or 'close' in col.lower()]
        for col in price_columns:
            if (data[col] < 0).any():
                issues.append(f"Negative values found in price column '{col}'")
        
        # Check for volume consistency
        if 'volume' in data.columns:
            if (data['volume'] < 0).any():
                issues.append("Negative volume values found")
        
        # Check timestamp ordering (if timestamp column exists)
        timestamp_columns = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        for col in timestamp_columns:
            if not data[col].is_monotonic_increasing:
                issues.append(f"Timestamp column '{col}' is not monotonically increasing")
        
        return issues


class StepDependencyValidator(BaseValidator):
    """Validates step dependencies and prerequisites."""
    
    def __init__(self):
        super().__init__("step_dependency", ValidationLevel.CRITICAL)
    
    @handles_errors(FileNotFoundError, ValueError, fallback=None)
    async def validate(self, step_name: str, context: Dict[str, Any]) -> ValidationReport:
        """Validate step dependencies."""
        start_time = time.time()
        
        try:
            warnings = []
            errors = []
            
            # Get step dependencies
            dependencies = self._get_step_dependencies(step_name)
            
            # Check if prerequisite files exist
            for dep_step, dep_files in dependencies.items():
                for file_path in dep_files:
                    if not safe_file_exists(file_path):
                        errors.append(f"Missing dependency file from {dep_step}: {file_path}")
            
            # Check if prerequisite steps completed successfully
            completed_steps = context.get("completed_steps", [])
            missing_steps = [dep for dep in dependencies.keys() if dep not in completed_steps]
            if missing_steps:
                errors.append(f"Missing prerequisite steps: {missing_steps}")
            
            # Check data freshness
            freshness_issues = await self._check_data_freshness(dependencies, context)
            warnings.extend(freshness_issues)
            
            result = ValidationResult.PASSED if not errors else ValidationResult.FAILED
            if warnings and not errors:
                result = ValidationResult.WARNING
            
            return self._create_report(
                result,
                f"Step dependency validation {'passed' if result == ValidationResult.PASSED else 'failed'}",
                details={"dependencies": dependencies, "completed_steps": completed_steps},
                warnings=warnings,
                errors=errors,
                execution_time=time.time() - start_time
            )
        
        except Exception as e:
            return self._create_report(
                ValidationResult.FAILED,
                f"Step dependency validation failed: {str(e)}",
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    def _get_step_dependencies(self, step_name: str) -> Dict[str, List[str]]:
        """Get dependencies for a specific step."""
        # Define step dependencies
        dependencies = {
            "step1_data_collection": {},
            "step1_5_data_converter": {
                "step1_data_collection": [
                    "data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet",
                    "data_cache/volume_BINANCE_ETHUSDT_consolidated.parquet"
                ]
            },
            "step2_data_reading": {
                "step1_5_data_converter": [
                    "data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet"
                ]
            },
            "step2_5_sr_optimization": {
                "step2_data_reading": [
                    "data_cache/processed_data.parquet"
                ]
            },
            "step3_hmm_regime_discovery": {
                "step2_5_sr_optimization": [
                    "data_cache/sr_optimized_data.parquet"
                ]
            },
            "step3_5_final_regime_clustering": {
                "step3_hmm_regime_discovery": [
                    "data_cache/regime_data.parquet"
                ]
            },
            "step4_regime_data_splitting": {
                "step3_5_final_regime_clustering": [
                    "data_cache/final_regime_data.parquet"
                ]
            },
            "step5_labeling": {
                "step4_regime_data_splitting": [
                    "data_cache/regime_split_data.parquet"
                ]
            },
            "step6_feature_engineering": {
                "step5_labeling": [
                    "data_cache/labeled_data.parquet"
                ]
            },
            "step7_regime_data_splitting": {
                "step6_feature_engineering": [
                    "data_cache/feature_engineered_data.parquet"
                ]
            },
            "step8_hmm_based_training": {
                "step7_regime_data_splitting": [
                    "data_cache/regime_split_features.parquet"
                ]
            },
            "step9_hmm_based_training": {
                "step8_hmm_based_training": [
                    "models/hmm_models.pkl"
                ]
            },
            "step10_unified_regime_intelligence": {
                "step9_hmm_based_training": [
                    "models/regime_intelligence.pkl"
                ]
            },
            "step11_analyst_creation": {
                "step10_unified_regime_intelligence": [
                    "models/regime_intelligence.pkl"
                ]
            },
            "step12_analyst_enhancement": {
                "step11_analyst_creation": [
                    "models/analyst_models.pkl"
                ]
            },
            "step13_analyst_ensemble_creation": {
                "step12_analyst_enhancement": [
                    "models/enhanced_analyst_models.pkl"
                ]
            },
            "step14_tactician_labeling": {
                "step13_analyst_ensemble_creation": [
                    "models/ensemble_models.pkl"
                ]
            },
            "step15_tactician_specialist_training": {
                "step14_tactician_labeling": [
                    "data_cache/tactician_labels.parquet"
                ]
            },
            "step16_confidence_calibration": {
                "step15_tactician_specialist_training": [
                    "models/tactician_models.pkl"
                ]
            },
            "step17_final_parameters_optimization": {
                "step16_confidence_calibration": [
                    "models/calibrated_models.pkl"
                ]
            },
            "step18_walk_forward_validation": {
                "step17_final_parameters_optimization": [
                    "models/optimized_models.pkl"
                ]
            },
            "step19_monte_carlo_validation": {
                "step18_walk_forward_validation": [
                    "models/validated_models.pkl"
                ]
            },
            "step20_ab_testing": {
                "step19_monte_carlo_validation": [
                    "models/monte_carlo_models.pkl"
                ]
            },
            "step21_saving": {
                "step20_ab_testing": [
                    "models/ab_tested_models.pkl"
                ]
            }
        }
        
        return dependencies.get(step_name, {})
    
    async def _check_data_freshness(self, dependencies: Dict[str, List[str]], context: Dict[str, Any]) -> List[str]:
        """Check if dependency data is fresh enough."""
        issues = []
        max_age_hours = context.get("max_data_age_hours", 24)
        
        for step_name, files in dependencies.items():
            for file_path in files:
                if safe_file_exists(file_path):
                    file_stat = Path(file_path).stat()
                    file_age_hours = (time.time() - file_stat.st_mtime) / 3600
                    
                    if file_age_hours > max_age_hours:
                        issues.append(f"Data file {file_path} is {file_age_hours:.1f} hours old (max: {max_age_hours}h)")
        
        return issues


class PipelineValidatorOrchestrator:
    """Orchestrates all pipeline validations."""
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline_validator_orchestrator")
        self.validators = {
            "data_format": DataFormatValidator(),
            "data_quality": DataQualityValidator(),
            "step_dependency": StepDependencyValidator()
        }
        self.validation_history = []
    
    @handles_errors(Exception, fallback=False)
    @logs_execution("pipeline_validation")
    async def validate_pipeline_step(
        self,
        step_name: str,
        data: Any,
        context: Dict[str, Any],
        validators_to_run: Optional[List[str]] = None
    ) -> Dict[str, ValidationReport]:
        """Validate a pipeline step with multiple validators."""
        
        if validators_to_run is None:
            validators_to_run = list(self.validators.keys())
        
        results = {}
        
        for validator_name in validators_to_run:
            if validator_name not in self.validators:
                self.logger.warning(f"Unknown validator: {validator_name}")
                continue
            
            validator = self.validators[validator_name]
            
            try:
                if validator_name == "step_dependency":
                    report = await validator.validate(step_name, context)
                else:
                    report = await validator.validate(data, context)
                
                results[validator_name] = report
                self.validation_history.append(report)
                
                # Log validation result
                if report.result == ValidationResult.FAILED:
                    self.logger.error(f"Validation failed for {step_name}: {report.message}")
                elif report.result == ValidationResult.WARNING:
                    self.logger.warning(f"Validation warning for {step_name}: {report.message}")
                else:
                    self.logger.info(f"Validation passed for {step_name}: {report.message}")
                
            except Exception as e:
                self.logger.exception(f"Validator {validator_name} failed for {step_name}: {e}")
                results[validator_name] = ValidationReport(
                    step_name=step_name,
                    validation_level=ValidationLevel.CRITICAL,
                    result=ValidationResult.FAILED,
                    message=f"Validator {validator_name} failed: {str(e)}",
                    errors=[str(e)]
                )
        
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed."""
        if not self.validation_history:
            return {"message": "No validations performed"}
        
        total_validations = len(self.validation_history)
        passed = sum(1 for r in self.validation_history if r.result == ValidationResult.PASSED)
        failed = sum(1 for r in self.validation_history if r.result == ValidationResult.FAILED)
        warnings = sum(1 for r in self.validation_history if r.result == ValidationResult.WARNING)
        
        return {
            "total_validations": total_validations,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "success_rate": passed / total_validations if total_validations > 0 else 0,
            "recent_validations": [r.to_dict() for r in self.validation_history[-10:]]
        }
    
    def save_validation_report(self, file_path: str) -> None:
        """Save validation history to file."""
        report_data = {
            "summary": self.get_validation_summary(),
            "validation_history": [r.to_dict() for r in self.validation_history],
            "timestamp": format_datetime(get_current_datetime())
        }
        
        safe_json_dump(report_data, file_path, indent=2)
        self.logger.info(f"Validation report saved to {file_path}")


# Global validator orchestrator instance
validator_orchestrator = PipelineValidatorOrchestrator()