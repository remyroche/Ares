"""
Feature Validation System

This module provides comprehensive validation and quality assurance
for the unified feature generation system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import json

from .core import FeatureGenerator, FeatureGenerationResult, FeatureCategory
from ...utils.logger import system_logger
from ...core.decorators import handles_errors


@dataclass
class ValidationRule:
    """A validation rule for features."""
    name: str
    description: str
    check_function: callable
    severity: str = "error"  # "error", "warning", "info"
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of feature validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    validated_at: datetime = field(default_factory=datetime.now)


@dataclass
class QualityMetrics:
    """Quality metrics for features."""
    completeness: float = 0.0
    consistency: float = 0.0
    stability: float = 0.0
    performance: float = 0.0
    overall_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class FeatureValidator:
    """
    Validator for feature generation results.
    
    Provides comprehensive validation including data quality,
    consistency, and performance checks.
    """
    
    def __init__(self):
        """Initialize the feature validator."""
        self.logger = system_logger.getChild("FeatureValidator")
        self._validation_rules: Dict[str, ValidationRule] = {}
        self._initialized = False
        
        # Register default validation rules
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """Register default validation rules."""
        # Data completeness rules
        self._add_rule(ValidationRule(
            name="no_nan_values",
            description="Check for NaN values in features",
            check_function=self._check_no_nan_values,
            severity="error"
        ))
        
        self._add_rule(ValidationRule(
            name="no_infinite_values",
            description="Check for infinite values in features",
            check_function=self._check_no_infinite_values,
            severity="error"
        ))
        
        self._add_rule(ValidationRule(
            name="no_duplicate_columns",
            description="Check for duplicate column names",
            check_function=self._check_no_duplicate_columns,
            severity="error"
        ))
        
        # Data quality rules
        self._add_rule(ValidationRule(
            name="reasonable_value_ranges",
            description="Check for reasonable value ranges",
            check_function=self._check_reasonable_ranges,
            severity="warning"
        ))
        
        self._add_rule(ValidationRule(
            name="sufficient_variance",
            description="Check for sufficient variance in features",
            check_function=self._check_sufficient_variance,
            severity="warning"
        ))
        
        # Performance rules
        self._add_rule(ValidationRule(
            name="generation_time_reasonable",
            description="Check if generation time is reasonable",
            check_function=self._check_generation_time,
            severity="info"
        ))
        
        self._add_rule(ValidationRule(
            name="memory_usage_reasonable",
            description="Check if memory usage is reasonable",
            check_function=self._check_memory_usage,
            severity="info"
        ))
    
    def _add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self._validation_rules[rule.name] = rule
    
    async def initialize(self) -> bool:
        """Initialize the validator."""
        try:
            self.logger.info("Initializing feature validator...")
            self._initialized = True
            self.logger.info(f"Feature validator initialized with {len(self._validation_rules)} rules")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing validator: {e}")
            return False
    
    @handles_errors(exceptions=(Exception,), default_return=ValidationResult(is_valid=False), context="feature validation")
    async def validate_features(
        self,
        result: FeatureGenerationResult,
        generator: Optional[FeatureGenerator] = None,
        custom_rules: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate feature generation result.
        
        Args:
            result: Feature generation result to validate
            generator: Optional generator that produced the result
            custom_rules: Optional list of specific rules to run
            
        Returns:
            ValidationResult with validation details
        """
        if not self._initialized:
            return ValidationResult(
                is_valid=False,
                errors=["Validator not initialized"]
            )
        
        try:
            errors = []
            warnings = []
            info = []
            metrics = {}
            
            # Check if result is valid
            if not result.success:
                errors.append("Feature generation was not successful")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    info=info,
                    metrics=metrics
                )
            
            if result.features is None or result.features.empty:
                errors.append("No features generated")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    info=info,
                    metrics=metrics
                )
            
            # Run validation rules
            rules_to_run = custom_rules if custom_rules else list(self._validation_rules.keys())
            
            for rule_name in rules_to_run:
                rule = self._validation_rules.get(rule_name)
                if not rule or not rule.enabled:
                    continue
                
                try:
                    rule_result = await self._run_validation_rule(rule, result, generator)
                    
                    if rule_result["passed"]:
                        if rule.severity == "info":
                            info.append(f"{rule.name}: {rule_result.get('message', 'OK')}")
                    else:
                        message = rule_result.get("message", f"Rule {rule.name} failed")
                        
                        if rule.severity == "error":
                            errors.append(f"{rule.name}: {message}")
                        elif rule.severity == "warning":
                            warnings.append(f"{rule.name}: {message}")
                        else:
                            info.append(f"{rule.name}: {message}")
                    
                    # Collect metrics
                    if "metrics" in rule_result:
                        metrics.update(rule_result["metrics"])
                        
                except Exception as e:
                    self.logger.warning(f"Error running validation rule {rule_name}: {e}")
                    warnings.append(f"Rule {rule_name} failed to run: {str(e)}")
            
            # Calculate overall validity
            is_valid = len(errors) == 0
            
            # Add summary metrics
            metrics.update({
                "total_errors": len(errors),
                "total_warnings": len(warnings),
                "total_info": len(info),
                "validation_rules_run": len(rules_to_run),
                "is_valid": is_valid
            })
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                info=info,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error in feature validation: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"]
            )
    
    async def _run_validation_rule(
        self,
        rule: ValidationRule,
        result: FeatureGenerationResult,
        generator: Optional[FeatureGenerator]
    ) -> Dict[str, Any]:
        """Run a single validation rule."""
        try:
            if asyncio.iscoroutinefunction(rule.check_function):
                return await rule.check_function(result, generator, rule.parameters)
            else:
                return rule.check_function(result, generator, rule.parameters)
        except Exception as e:
            return {
                "passed": False,
                "message": f"Rule execution error: {str(e)}"
            }
    
    def _check_no_nan_values(self, result: FeatureGenerationResult, generator: Optional[FeatureGenerator], params: Dict[str, Any]) -> Dict[str, Any]:
        """Check for NaN values."""
        features = result.features
        nan_columns = features.columns[features.isnull().any()].tolist()
        
        if nan_columns:
            return {
                "passed": False,
                "message": f"NaN values found in columns: {nan_columns}",
                "metrics": {"nan_columns": nan_columns}
            }
        else:
            return {"passed": True, "message": "No NaN values found"}
    
    def _check_no_infinite_values(self, result: FeatureGenerationResult, generator: Optional[FeatureGenerator], params: Dict[str, Any]) -> Dict[str, Any]:
        """Check for infinite values."""
        features = result.features
        inf_columns = features.columns[np.isinf(features).any()].tolist()
        
        if inf_columns:
            return {
                "passed": False,
                "message": f"Infinite values found in columns: {inf_columns}",
                "metrics": {"inf_columns": inf_columns}
            }
        else:
            return {"passed": True, "message": "No infinite values found"}
    
    def _check_no_duplicate_columns(self, result: FeatureGenerationResult, generator: Optional[FeatureGenerator], params: Dict[str, Any]) -> Dict[str, Any]:
        """Check for duplicate column names."""
        features = result.features
        duplicate_columns = features.columns[features.columns.duplicated()].tolist()
        
        if duplicate_columns:
            return {
                "passed": False,
                "message": f"Duplicate columns found: {duplicate_columns}",
                "metrics": {"duplicate_columns": duplicate_columns}
            }
        else:
            return {"passed": True, "message": "No duplicate columns found"}
    
    def _check_reasonable_ranges(self, result: FeatureGenerationResult, generator: Optional[FeatureGenerator], params: Dict[str, Any]) -> Dict[str, Any]:
        """Check for reasonable value ranges."""
        features = result.features
        numeric_features = features.select_dtypes(include=[np.number])
        
        if numeric_features.empty:
            return {"passed": True, "message": "No numeric features to check"}
        
        extreme_columns = []
        for col in numeric_features.columns:
            col_data = numeric_features[col].dropna()
            if len(col_data) > 0:
                # Check for extremely large values (beyond 6 standard deviations)
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                if (z_scores > 6).any():
                    extreme_columns.append(col)
        
        if extreme_columns:
            return {
                "passed": False,
                "message": f"Extreme values found in columns: {extreme_columns}",
                "metrics": {"extreme_columns": extreme_columns}
            }
        else:
            return {"passed": True, "message": "All values within reasonable ranges"}
    
    def _check_sufficient_variance(self, result: FeatureGenerationResult, generator: Optional[FeatureGenerator], params: Dict[str, Any]) -> Dict[str, Any]:
        """Check for sufficient variance in features."""
        features = result.features
        numeric_features = features.select_dtypes(include=[np.number])
        
        if numeric_features.empty:
            return {"passed": True, "message": "No numeric features to check"}
        
        low_variance_columns = []
        for col in numeric_features.columns:
            col_data = numeric_features[col].dropna()
            if len(col_data) > 1:
                variance = col_data.var()
                if variance < 1e-10:  # Very low variance threshold
                    low_variance_columns.append(col)
        
        if low_variance_columns:
            return {
                "passed": False,
                "message": f"Low variance features found: {low_variance_columns}",
                "metrics": {"low_variance_columns": low_variance_columns}
            }
        else:
            return {"passed": True, "message": "All features have sufficient variance"}
    
    def _check_generation_time(self, result: FeatureGenerationResult, generator: Optional[FeatureGenerator], params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if generation time is reasonable."""
        max_time = params.get("max_time_seconds", 30.0)
        
        if "duration_seconds" in result.performance_metrics:
            duration = result.performance_metrics["duration_seconds"]
            if duration > max_time:
                return {
                    "passed": False,
                    "message": f"Generation time {duration:.2f}s exceeds limit {max_time}s",
                    "metrics": {"duration_seconds": duration, "max_time_seconds": max_time}
                }
            else:
                return {
                    "passed": True,
                    "message": f"Generation time {duration:.2f}s is reasonable",
                    "metrics": {"duration_seconds": duration}
                }
        else:
            return {"passed": True, "message": "No timing information available"}
    
    def _check_memory_usage(self, result: FeatureGenerationResult, generator: Optional[FeatureGenerator], params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if memory usage is reasonable."""
        max_memory_mb = params.get("max_memory_mb", 1000.0)
        
        if result.features is not None:
            memory_usage = result.features.memory_usage(deep=True).sum() / 1024 / 1024  # Convert to MB
            
            if memory_usage > max_memory_mb:
                return {
                    "passed": False,
                    "message": f"Memory usage {memory_usage:.2f}MB exceeds limit {max_memory_mb}MB",
                    "metrics": {"memory_usage_mb": memory_usage, "max_memory_mb": max_memory_mb}
                }
            else:
                return {
                    "passed": True,
                    "message": f"Memory usage {memory_usage:.2f}MB is reasonable",
                    "metrics": {"memory_usage_mb": memory_usage}
                }
        else:
            return {"passed": True, "message": "No features to check memory usage"}
    
    def add_validation_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule."""
        self._validation_rules[rule.name] = rule
        self.logger.info(f"Added validation rule: {rule.name}")
    
    def remove_validation_rule(self, rule_name: str) -> bool:
        """Remove a validation rule."""
        if rule_name in self._validation_rules:
            del self._validation_rules[rule_name]
            self.logger.info(f"Removed validation rule: {rule_name}")
            return True
        return False
    
    def list_validation_rules(self) -> List[str]:
        """List all validation rules."""
        return list(self._validation_rules.keys())
    
    def get_validation_rule(self, rule_name: str) -> Optional[ValidationRule]:
        """Get a validation rule by name."""
        return self._validation_rules.get(rule_name)


class FeatureConsistencyChecker:
    """
    Checker for feature consistency across different runs and generators.
    """
    
    def __init__(self):
        """Initialize the consistency checker."""
        self.logger = system_logger.getChild("FeatureConsistencyChecker")
        self._baseline_features: Optional[pd.DataFrame] = None
        self._baseline_metadata: Dict[str, Any] = {}
    
    def set_baseline(self, features: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Set baseline features for consistency checking."""
        self._baseline_features = features.copy()
        self._baseline_metadata = metadata or {}
        self.logger.info("Baseline features set for consistency checking")
    
    @handles_errors(exceptions=(Exception,), default_return=False, context="consistency check")
    async def check_consistency(
        self,
        current_features: pd.DataFrame,
        tolerance: float = 1e-6
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check consistency with baseline features.
        
        Args:
            current_features: Current features to check
            tolerance: Tolerance for numerical differences
            
        Returns:
            Tuple of (is_consistent, details)
        """
        if self._baseline_features is None:
            return False, {"error": "No baseline features set"}
        
        try:
            details = {}
            
            # Check column consistency
            baseline_cols = set(self._baseline_features.columns)
            current_cols = set(current_features.columns)
            
            missing_cols = baseline_cols - current_cols
            extra_cols = current_cols - baseline_cols
            
            details["missing_columns"] = list(missing_cols)
            details["extra_columns"] = list(extra_cols)
            details["column_consistency"] = len(missing_cols) == 0 and len(extra_cols) == 0
            
            # Check shape consistency
            details["shape_consistency"] = (
                self._baseline_features.shape == current_features.shape
            )
            details["baseline_shape"] = self._baseline_features.shape
            details["current_shape"] = current_features.shape
            
            # Check numerical consistency for common columns
            common_cols = baseline_cols & current_cols
            numerical_consistency = True
            
            for col in common_cols:
                if col in self._baseline_features.columns and col in current_features.columns:
                    baseline_data = self._baseline_features[col].dropna()
                    current_data = current_features[col].dropna()
                    
                    if len(baseline_data) > 0 and len(current_data) > 0:
                        # Check if data is numerically similar
                        if baseline_data.dtype in [np.float64, np.float32, np.int64, np.int32]:
                            max_diff = np.abs(baseline_data - current_data).max()
                            if max_diff > tolerance:
                                numerical_consistency = False
                                break
            
            details["numerical_consistency"] = numerical_consistency
            
            # Overall consistency
            is_consistent = (
                details["column_consistency"] and
                details["shape_consistency"] and
                details["numerical_consistency"]
            )
            
            details["is_consistent"] = is_consistent
            
            return is_consistent, details
            
        except Exception as e:
            self.logger.error(f"Error in consistency check: {e}")
            return False, {"error": str(e)}


class FeatureQualityMetrics:
    """
    Calculator for feature quality metrics.
    """
    
    def __init__(self):
        """Initialize the quality metrics calculator."""
        self.logger = system_logger.getChild("FeatureQualityMetrics")
    
    @handles_errors(exceptions=(Exception,), default_return=QualityMetrics(), context="quality metrics calculation")
    async def calculate_quality_metrics(
        self,
        result: FeatureGenerationResult,
        generator: Optional[FeatureGenerator] = None
    ) -> QualityMetrics:
        """
        Calculate quality metrics for feature generation result.
        
        Args:
            result: Feature generation result
            generator: Optional generator that produced the result
            
        Returns:
            QualityMetrics object
        """
        try:
            if not result.success or result.features is None:
                return QualityMetrics()
            
            features = result.features
            metrics = QualityMetrics()
            
            # Calculate completeness
            metrics.completeness = self._calculate_completeness(features)
            
            # Calculate consistency
            metrics.consistency = self._calculate_consistency(features)
            
            # Calculate stability
            metrics.stability = self._calculate_stability(features)
            
            # Calculate performance
            metrics.performance = self._calculate_performance(result)
            
            # Calculate overall score
            metrics.overall_score = (
                metrics.completeness * 0.3 +
                metrics.consistency * 0.3 +
                metrics.stability * 0.2 +
                metrics.performance * 0.2
            )
            
            # Add detailed metrics
            metrics.details = {
                "feature_count": len(features.columns),
                "row_count": len(features),
                "memory_usage_mb": features.memory_usage(deep=True).sum() / 1024 / 1024,
                "numeric_features": len(features.select_dtypes(include=[np.number]).columns),
                "categorical_features": len(features.select_dtypes(include=['object', 'category']).columns)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {e}")
            return QualityMetrics()
    
    def _calculate_completeness(self, features: pd.DataFrame) -> float:
        """Calculate completeness score."""
        if features.empty:
            return 0.0
        
        # Calculate percentage of non-null values
        total_values = features.size
        null_values = features.isnull().sum().sum()
        completeness = (total_values - null_values) / total_values
        
        return float(completeness)
    
    def _calculate_consistency(self, features: pd.DataFrame) -> float:
        """Calculate consistency score."""
        if features.empty:
            return 0.0
        
        numeric_features = features.select_dtypes(include=[np.number])
        if numeric_features.empty:
            return 1.0  # No numeric features to check
        
        # Check for consistent data types and ranges
        consistency_score = 1.0
        
        for col in numeric_features.columns:
            col_data = numeric_features[col].dropna()
            if len(col_data) > 1:
                # Check for reasonable variance
                variance = col_data.var()
                if variance < 1e-10:  # Very low variance
                    consistency_score -= 0.1
                
                # Check for extreme outliers
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                extreme_outliers = (z_scores > 6).sum()
                if extreme_outliers > len(col_data) * 0.01:  # More than 1% extreme outliers
                    consistency_score -= 0.1
        
        return max(0.0, consistency_score)
    
    def _calculate_stability(self, features: pd.DataFrame) -> float:
        """Calculate stability score."""
        if features.empty:
            return 0.0
        
        # For now, use a simple stability metric
        # In practice, this would compare with historical data
        numeric_features = features.select_dtypes(include=[np.number])
        if numeric_features.empty:
            return 1.0
        
        # Calculate coefficient of variation for each feature
        stability_scores = []
        for col in numeric_features.columns:
            col_data = numeric_features[col].dropna()
            if len(col_data) > 1 and col_data.mean() != 0:
                cv = col_data.std() / abs(col_data.mean())
                # Lower CV is more stable
                stability_score = max(0.0, 1.0 - cv)
                stability_scores.append(stability_score)
        
        return np.mean(stability_scores) if stability_scores else 1.0
    
    def _calculate_performance(self, result: FeatureGenerationResult) -> float:
        """Calculate performance score."""
        if "duration_seconds" not in result.performance_metrics:
            return 1.0  # No timing information
        
        duration = result.performance_metrics["duration_seconds"]
        
        # Performance score based on duration (lower is better)
        if duration <= 1.0:
            return 1.0
        elif duration <= 5.0:
            return 0.8
        elif duration <= 10.0:
            return 0.6
        elif duration <= 30.0:
            return 0.4
        else:
            return 0.2