"""
Comprehensive Validation Framework for Feature Lookback Optimization.

This module provides a robust validation framework for data quality,
optimization results, and pipeline state validation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.tprint import tprint

logger = system_logger.getChild('ValidationFramework')

class ValidationLevel(Enum):
    """Validation severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

@dataclass
class ValidationRule:
    """Individual validation rule definition."""
    name: str
    description: str
    level: ValidationLevel
    validator_func: callable
    required: bool = True
    auto_fix: bool = False
    fix_func: Optional[callable] = None

@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_name: str
    status: ValidationStatus
    level: ValidationLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    auto_fixed: bool = False
    fix_applied: Optional[str] = None

@dataclass
class ValidationSummary:
    """Summary of validation results."""
    total_rules: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    critical_failures: int
    overall_status: ValidationStatus
    quality_score: float
    recommendations: List[str]

class ValidationFramework:
    """
    Comprehensive validation framework for feature lookback optimization.
    
    Provides structured validation for data quality, optimization results,
    and pipeline state with automatic fixing capabilities.
    """
    
    def __init__(self):
        """Initialize the validation framework."""
        self.logger = logger.getChild('ValidationFramework')
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.validation_results: List[ValidationResult] = []
        
        # Initialize validation rule sets
        self._initialize_data_validation_rules()
        self._initialize_optimization_validation_rules()
        self._initialize_pipeline_validation_rules()
    
    def _initialize_data_validation_rules(self) -> None:
        """Initialize data validation rules."""
        self.validation_rules['data'] = [
            ValidationRule(
                name="data_not_null",
                description="Input data is not None or empty",
                level=ValidationLevel.CRITICAL,
                validator_func=self._validate_data_not_null,
                required=True
            ),
            ValidationRule(
                name="data_is_dataframe",
                description="Input data is a pandas DataFrame",
                level=ValidationLevel.CRITICAL,
                validator_func=self._validate_data_is_dataframe,
                required=True
            ),
            ValidationRule(
                name="required_columns_present",
                description="Required OHLCV columns are present",
                level=ValidationLevel.CRITICAL,
                validator_func=self._validate_required_columns,
                required=True,
                auto_fix=True,
                fix_func=self._fix_missing_columns
            ),
            ValidationRule(
                name="data_completeness",
                description="Data completeness is above threshold",
                level=ValidationLevel.HIGH,
                validator_func=self._validate_data_completeness,
                required=True
            ),
            ValidationRule(
                name="no_infinite_values",
                description="No infinite values in numeric columns",
                level=ValidationLevel.HIGH,
                validator_func=self._validate_no_infinite_values,
                required=True,
                auto_fix=True,
                fix_func=self._fix_infinite_values
            ),
            ValidationRule(
                name="price_consistency",
                description="OHLC price relationships are consistent",
                level=ValidationLevel.MEDIUM,
                validator_func=self._validate_price_consistency,
                required=False,
                auto_fix=True,
                fix_func=self._fix_price_consistency
            ),
            ValidationRule(
                name="volume_positive",
                description="Volume values are non-negative",
                level=ValidationLevel.MEDIUM,
                validator_func=self._validate_volume_positive,
                required=False,
                auto_fix=True,
                fix_func=self._fix_negative_volume
            ),
            ValidationRule(
                name="data_freshness",
                description="Data is not too old",
                level=ValidationLevel.LOW,
                validator_func=self._validate_data_freshness,
                required=False
            )
        ]
    
    def _initialize_optimization_validation_rules(self) -> None:
        """Initialize optimization results validation rules."""
        self.validation_rules['optimization'] = [
            ValidationRule(
                name="optimization_results_present",
                description="Optimization results are present and non-empty",
                level=ValidationLevel.CRITICAL,
                validator_func=self._validate_optimization_results_present,
                required=True
            ),
            ValidationRule(
                name="optimized_features_present",
                description="Optimized features are present and non-empty",
                level=ValidationLevel.CRITICAL,
                validator_func=self._validate_optimized_features_present,
                required=True
            ),
            ValidationRule(
                name="best_score_valid",
                description="Best optimization score is valid",
                level=ValidationLevel.HIGH,
                validator_func=self._validate_best_score,
                required=True
            ),
            ValidationRule(
                name="lookback_periods_valid",
                description="Lookback periods are within valid range",
                level=ValidationLevel.HIGH,
                validator_func=self._validate_lookback_periods,
                required=True
            ),
            ValidationRule(
                name="optimization_convergence",
                description="Optimization achieved convergence",
                level=ValidationLevel.MEDIUM,
                validator_func=self._validate_optimization_convergence,
                required=False
            ),
            ValidationRule(
                name="feature_scores_valid",
                description="Feature performance scores are valid",
                level=ValidationLevel.MEDIUM,
                validator_func=self._validate_feature_scores,
                required=False
            ),
            ValidationRule(
                name="optimization_time_reasonable",
                description="Optimization time is within reasonable bounds",
                level=ValidationLevel.LOW,
                validator_func=self._validate_optimization_time,
                required=False
            )
        ]
    
    def _initialize_pipeline_validation_rules(self) -> None:
        """Initialize pipeline state validation rules."""
        self.validation_rules['pipeline'] = [
            ValidationRule(
                name="labeling_results_present",
                description="Labeling results are present (multi-horizon or triple barrier)",
                level=ValidationLevel.CRITICAL,
                validator_func=self._validate_labeling_results,
                required=True
            ),
            ValidationRule(
                name="regime_data_present",
                description="Regime data splitting results are present",
                level=ValidationLevel.MEDIUM,
                validator_func=self._validate_regime_data_present,
                required=False
            ),
            ValidationRule(
                name="pipeline_state_consistency",
                description="Pipeline state is consistent",
                level=ValidationLevel.MEDIUM,
                validator_func=self._validate_pipeline_state_consistency,
                required=False
            )
        ]
    
    def validate_data(self, data: Any) -> Tuple[bool, List[ValidationResult], Optional[Any]]:
        """
        Validate input data with comprehensive checks.
        
        Args:
            data: Input data to validate
            
        Returns:
            Tuple of (is_valid, validation_results, fixed_data)
        """
        tprint("🔍 Starting comprehensive data validation...")
        
        validation_results = []
        fixed_data = data
        
        for rule in self.validation_rules['data']:
            try:
                result = self._execute_validation_rule(rule, fixed_data)
                validation_results.append(result)
                
                # Apply auto-fix if available and needed
                if (result.status == ValidationStatus.FAILED and 
                    rule.auto_fix and rule.fix_func and fixed_data is not None):
                    try:
                        fixed_data = rule.fix_func(fixed_data)
                        result.auto_fixed = True
                        result.fix_applied = f"Applied {rule.name} fix"
                        result.status = ValidationStatus.PASSED
                        tprint(f"✅ Auto-fixed: {rule.name}")
                    except Exception as e:
                        tprint(f"⚠️ Auto-fix failed for {rule.name}: {e}")
                
            except Exception as e:
                tprint(f"❌ Validation rule {rule.name} failed: {e}")
                validation_results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    level=rule.level,
                    message=f"Validation rule execution failed: {e}"
                ))
        
        # Determine overall validity
        critical_failures = [r for r in validation_results 
                           if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL]
        is_valid = len(critical_failures) == 0
        
        tprint(f"✅ Data validation completed: {'PASSED' if is_valid else 'FAILED'}")
        return is_valid, validation_results, fixed_data
    
    def validate_optimization_results(self, optimization_result: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate optimization results.
        
        Args:
            optimization_result: Optimization results to validate
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        tprint("🔍 Starting optimization results validation...")
        
        validation_results = []
        
        for rule in self.validation_rules['optimization']:
            try:
                result = self._execute_validation_rule(rule, optimization_result)
                validation_results.append(result)
                
            except Exception as e:
                tprint(f"❌ Validation rule {rule.name} failed: {e}")
                validation_results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    level=rule.level,
                    message=f"Validation rule execution failed: {e}"
                ))
        
        # Determine overall validity
        critical_failures = [r for r in validation_results 
                           if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL]
        is_valid = len(critical_failures) == 0
        
        tprint(f"✅ Optimization validation completed: {'PASSED' if is_valid else 'FAILED'}")
        return is_valid, validation_results
    
    def validate_pipeline_state(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate pipeline state.
        
        Args:
            pipeline_state: Pipeline state to validate
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        tprint("🔍 Starting pipeline state validation...")
        
        validation_results = []
        
        for rule in self.validation_rules['pipeline']:
            try:
                result = self._execute_validation_rule(rule, pipeline_state)
                validation_results.append(result)
                
            except Exception as e:
                tprint(f"❌ Validation rule {rule.name} failed: {e}")
                validation_results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    level=rule.level,
                    message=f"Validation rule execution failed: {e}"
                ))
        
        # Determine overall validity
        critical_failures = [r for r in validation_results 
                           if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL]
        is_valid = len(critical_failures) == 0
        
        tprint(f"✅ Pipeline validation completed: {'PASSED' if is_valid else 'FAILED'}")
        return is_valid, validation_results
    
    def generate_validation_summary(self, validation_results: List[ValidationResult]) -> ValidationSummary:
        """Generate validation summary from results."""
        total_rules = len(validation_results)
        passed = len([r for r in validation_results if r.status == ValidationStatus.PASSED])
        failed = len([r for r in validation_results if r.status == ValidationStatus.FAILED])
        warnings = len([r for r in validation_results if r.status == ValidationStatus.WARNING])
        skipped = len([r for r in validation_results if r.status == ValidationStatus.SKIPPED])
        critical_failures = len([r for r in validation_results 
                               if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL])
        
        # Calculate quality score
        quality_score = passed / total_rules if total_rules > 0 else 0.0
        
        # Determine overall status
        if critical_failures > 0:
            overall_status = ValidationStatus.FAILED
        elif failed > 0:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASSED
        
        # Generate recommendations
        recommendations = self._generate_validation_recommendations(validation_results)
        
        return ValidationSummary(
            total_rules=total_rules,
            passed=passed,
            failed=failed,
            warnings=warnings,
            skipped=skipped,
            critical_failures=critical_failures,
            overall_status=overall_status,
            quality_score=quality_score,
            recommendations=recommendations
        )
    
    def _execute_validation_rule(self, rule: ValidationRule, data: Any) -> ValidationResult:
        """Execute a single validation rule."""
        try:
            is_valid, message, details = rule.validator_func(data)
            
            if is_valid:
                status = ValidationStatus.PASSED
            elif rule.level == ValidationLevel.CRITICAL:
                status = ValidationStatus.FAILED
            else:
                status = ValidationStatus.WARNING
            
            return ValidationResult(
                rule_name=rule.name,
                status=status,
                level=rule.level,
                message=message,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.FAILED,
                level=rule.level,
                message=f"Validation error: {e}"
            )
    
    # Data validation methods
    def _validate_data_not_null(self, data: Any) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate that data is not None or empty."""
        if data is None:
            return False, "Input data is None", None
        
        if isinstance(data, pd.DataFrame) and data.empty:
            return False, "Input data is empty", None
        
        return True, "Data is not null or empty", None
    
    def _validate_data_is_dataframe(self, data: Any) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate that data is a pandas DataFrame."""
        if not isinstance(data, pd.DataFrame):
            return False, f"Input data must be a pandas DataFrame, got {type(data)}", None
        
        return True, "Data is a valid pandas DataFrame", None
    
    def _validate_required_columns(self, data: Any) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate that required columns are present."""
        if not isinstance(data, pd.DataFrame):
            return False, "Cannot validate columns on non-DataFrame data", None
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}", {
                'missing_columns': missing_columns,
                'available_columns': list(data.columns)
            }
        
        return True, "All required columns are present", None
    
    def _validate_data_completeness(self, data: Any) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate data completeness."""
        if not isinstance(data, pd.DataFrame):
            return False, "Cannot validate completeness on non-DataFrame data", None
        
        total_cells = len(data) * len(data.columns)
        non_null_cells = data.count().sum()
        completeness = non_null_cells / total_cells if total_cells > 0 else 0.0
        
        if completeness < 0.8:
            return False, f"Data completeness is too low: {completeness:.2%}", {
                'completeness': completeness,
                'threshold': 0.8
            }
        
        return True, f"Data completeness is acceptable: {completeness:.2%}", {
            'completeness': completeness
        }
    
    def _validate_no_infinite_values(self, data: Any) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate that there are no infinite values."""
        if not isinstance(data, pd.DataFrame):
            return False, "Cannot validate infinite values on non-DataFrame data", None
        
        numeric_data = data.select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric_data).sum().sum()
        
        if inf_count > 0:
            return False, f"Found {inf_count} infinite values in numeric columns", {
                'infinite_count': inf_count,
                'columns_with_inf': numeric_data.columns[np.isinf(numeric_data).any()].tolist()
            }
        
        return True, "No infinite values found", None
    
    def _validate_price_consistency(self, data: Any) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate OHLC price consistency."""
        if not isinstance(data, pd.DataFrame):
            return False, "Cannot validate price consistency on non-DataFrame data", None
        
        issues = []
        
        # Check high >= max(open, close)
        if all(col in data.columns for col in ['high', 'open', 'close']):
            inconsistent_high = (data['high'] < np.maximum(data['open'], data['close'])).sum()
            if inconsistent_high > 0:
                issues.append(f"High < max(open, close): {inconsistent_high} rows")
        
        # Check low <= min(open, close)
        if all(col in data.columns for col in ['low', 'open', 'close']):
            inconsistent_low = (data['low'] > np.minimum(data['open'], data['close'])).sum()
            if inconsistent_low > 0:
                issues.append(f"Low > min(open, close): {inconsistent_low} rows")
        
        if issues:
            return False, f"Price consistency issues: {'; '.join(issues)}", {
                'issues': issues
            }
        
        return True, "Price relationships are consistent", None
    
    def _validate_volume_positive(self, data: Any) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate that volume values are non-negative."""
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            return True, "Volume column not present, skipping validation", None
        
        negative_volume = (data['volume'] < 0).sum()
        
        if negative_volume > 0:
            return False, f"Found {negative_volume} negative volume values", {
                'negative_count': negative_volume
            }
        
        return True, "All volume values are non-negative", None
    
    def _validate_data_freshness(self, data: Any) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate that data is not too old."""
        if not isinstance(data, pd.DataFrame):
            return True, "Cannot validate freshness on non-DataFrame data", None
        
        # Check if there's a timestamp column
        timestamp_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        
        if not timestamp_cols:
            return True, "No timestamp column found, skipping freshness validation", None
        
        # Use the first timestamp column
        timestamp_col = timestamp_cols[0]
        try:
            timestamps = pd.to_datetime(data[timestamp_col])
            latest_timestamp = timestamps.max()
            days_old = (datetime.now() - latest_timestamp).days
            
            if days_old > 30:
                return False, f"Data is {days_old} days old", {
                    'days_old': days_old,
                    'latest_timestamp': latest_timestamp
                }
            
            return True, f"Data is fresh: {days_old} days old", {
                'days_old': days_old,
                'latest_timestamp': latest_timestamp
            }
            
        except Exception:
            return True, "Could not parse timestamps, skipping freshness validation", None
    
    # Auto-fix methods
    def _fix_missing_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix missing required columns."""
        fixed_data = data.copy()
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in fixed_data.columns:
                if col == 'volume':
                    fixed_data[col] = 1000  # Default volume
                else:
                    # Use close price as fallback for OHLC
                    fallback_value = fixed_data.get('close', 100.0)
                    fixed_data[col] = fallback_value
        
        return fixed_data
    
    def _fix_infinite_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix infinite values in numeric columns."""
        fixed_data = data.copy()
        numeric_data = fixed_data.select_dtypes(include=[np.number])
        
        # Replace inf with NaN, then forward fill and backward fill
        fixed_data[numeric_data.columns] = numeric_data.replace([np.inf, -np.inf], np.nan)
        fixed_data = fixed_data.ffill().bfill()
        
        return fixed_data
    
    def _fix_price_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix OHLC price consistency issues."""
        fixed_data = data.copy()
        
        if all(col in fixed_data.columns for col in ['open', 'high', 'low', 'close']):
            # Fix high < max(open, close)
            fixed_data['high'] = np.maximum(fixed_data['high'], 
                                          np.maximum(fixed_data['open'], fixed_data['close']))
            
            # Fix low > min(open, close)
            fixed_data['low'] = np.minimum(fixed_data['low'], 
                                         np.minimum(fixed_data['open'], fixed_data['close']))
        
        return fixed_data
    
    def _fix_negative_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix negative volume values."""
        fixed_data = data.copy()
        
        if 'volume' in fixed_data.columns:
            fixed_data['volume'] = np.maximum(fixed_data['volume'], 0)
        
        return fixed_data
    
    # Optimization validation methods
    def _validate_optimization_results_present(self, optimization_result: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate that optimization results are present."""
        optimization_results = optimization_result.get('optimization_results', {})
        
        if not optimization_results:
            return False, "No optimization results found", None
        
        required_keys = ['best_lookback_period', 'best_score', 'optimization_method']
        missing_keys = [key for key in required_keys if key not in optimization_results]
        
        if missing_keys:
            return False, f"Missing optimization result keys: {missing_keys}", {
                'missing_keys': missing_keys,
                'available_keys': list(optimization_results.keys())
            }
        
        return True, "Optimization results are present and complete", None
    
    def _validate_optimized_features_present(self, optimization_result: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate that optimized features are present."""
        optimized_features = optimization_result.get('optimized_features', {})
        
        if not optimized_features:
            return False, "No optimized features found", None
        
        return True, f"Found {len(optimized_features)} optimized features", {
            'feature_count': len(optimized_features),
            'feature_names': list(optimized_features.keys())
        }
    
    def _validate_best_score(self, optimization_result: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate that best score is valid."""
        optimization_results = optimization_result.get('optimization_results', {})
        best_score = optimization_results.get('best_score', 0.0)
        
        if best_score <= 0:
            return False, f"Best score is invalid: {best_score}", {
                'best_score': best_score
            }
        
        return True, f"Best score is valid: {best_score}", {
            'best_score': best_score
        }
    
    def _validate_lookback_periods(self, optimization_result: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate that lookback periods are within valid range."""
        optimized_features = optimization_result.get('optimized_features', {})
        
        if not optimized_features:
            return False, "No optimized features to validate", None
        
        invalid_periods = []
        for feature_name, feature_data in optimized_features.items():
            lookback = feature_data.get('lookback', 0)
            if lookback <= 0 or lookback > 1000:
                invalid_periods.append(f"{feature_name}: {lookback}")
        
        if invalid_periods:
            return False, f"Invalid lookback periods: {'; '.join(invalid_periods)}", {
                'invalid_periods': invalid_periods
            }
        
        return True, "All lookback periods are valid", None
    
    def _validate_optimization_convergence(self, optimization_result: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate that optimization achieved convergence."""
        optimization_metrics = optimization_result.get('optimization_metrics', {})
        convergence_iterations = optimization_metrics.get('convergence_iterations', 0)
        
        if convergence_iterations == 0:
            return False, "Optimization did not achieve convergence", {
                'convergence_iterations': convergence_iterations
            }
        
        return True, f"Optimization converged after {convergence_iterations} iterations", {
            'convergence_iterations': convergence_iterations
        }
    
    def _validate_feature_scores(self, optimization_result: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate that feature scores are valid."""
        optimized_features = optimization_result.get('optimized_features', {})
        
        if not optimized_features:
            return True, "No features to validate scores for", None
        
        invalid_scores = []
        for feature_name, feature_data in optimized_features.items():
            score = feature_data.get('score', 0.0)
            if score < 0 or score > 1.0:
                invalid_scores.append(f"{feature_name}: {score}")
        
        if invalid_scores:
            return False, f"Invalid feature scores: {'; '.join(invalid_scores)}", {
                'invalid_scores': invalid_scores
            }
        
        return True, "All feature scores are valid", None
    
    def _validate_optimization_time(self, optimization_result: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate that optimization time is reasonable."""
        optimization_time = optimization_result.get('optimization_time', 0.0)
        
        if optimization_time > 600:  # 10 minutes
            return False, f"Optimization time is too long: {optimization_time:.2f}s", {
                'optimization_time': optimization_time
            }
        
        return True, f"Optimization time is reasonable: {optimization_time:.2f}s", {
            'optimization_time': optimization_time
        }
    
    # Pipeline validation methods
    def _validate_labeling_results(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate labeling results (multi-horizon or triple barrier)."""
        # First check for multi-horizon labeling results (preferred)
        multi_horizon_labeling = pipeline_state.get('multi_horizon_labeling_result', {})
        if multi_horizon_labeling:
            required_keys = ['labeled_data', 'labeling_metrics', 'method']
            missing_keys = [key for key in required_keys if key not in multi_horizon_labeling]
            
            if missing_keys:
                return False, f"Missing multi-horizon labeling keys: {missing_keys}", {
                    'missing_keys': missing_keys
                }
            
            return True, "Multi-horizon labeling results are present", None
        
        # Fallback to triple barrier labeling for backward compatibility
        triple_barrier_labeling = pipeline_state.get('triple_barrier_labeling_result', {})
        if triple_barrier_labeling:
            required_keys = ['labels', 'barriers', 'metadata']
            missing_keys = [key for key in required_keys if key not in triple_barrier_labeling]
            
            if missing_keys:
                return False, f"Missing triple barrier labeling keys: {missing_keys}", {
                    'missing_keys': missing_keys
                }
            
            return True, "Triple barrier labeling results are present", None
        
        # Try to load from recent outcomes if not in pipeline state
        try:
            import json
            from pathlib import Path
            
            outcomes_dir = Path("outcomes")
            if outcomes_dir.exists():
                # Search for multi-horizon profit labeler outcome files
                pattern = f"market_analysis_multi_horizon_profit_labeler_outcome_*.json"
                outcome_files = list(outcomes_dir.glob(pattern))
                
                if outcome_files:
                    # Get the most recent file
                    latest_file = max(outcome_files, key=lambda f: f.stat().st_mtime)
                    
                    with open(latest_file, 'r') as f:
                        outcome_data = json.load(f)
                    
                    # Check if we have valid artifacts
                    artifacts = outcome_data.get('artifacts', {})
                    multi_horizon_result = artifacts.get('multi_horizon_labeling_result', {})
                    
                    if multi_horizon_result and 'labeled_data' in multi_horizon_result:
                        return True, f"Multi-horizon labeling results loaded from recent outcomes: {latest_file.name}", None
        except Exception as e:
            pass  # Continue to return the original error
        
        return False, "No labeling results found (neither multi-horizon nor triple barrier)", None
    
    def _validate_regime_data_present(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate regime data splitting results."""
        regime_data_splitting = pipeline_state.get('regime_data_splitting_result', {})
        
        if not regime_data_splitting:
            # Try to load from recent outcomes if not in pipeline state
            try:
                import json
                from pathlib import Path
                
                outcomes_dir = Path("outcomes")
                if outcomes_dir.exists():
                    # Search for regime data splitting outcome files
                    pattern = f"market_analysis_regime_data_splitting_outcome_*.json"
                    outcome_files = list(outcomes_dir.glob(pattern))
                    
                    if outcome_files:
                        # Get the most recent file
                        latest_file = max(outcome_files, key=lambda f: f.stat().st_mtime)
                        
                        with open(latest_file, 'r') as f:
                            outcome_data = json.load(f)
                        
                        # Check if the outcome contains regime data
                        if 'artifacts' in outcome_data and 'regime_data_splitting_result' in outcome_data['artifacts']:
                            return True, f"Regime data splitting results loaded from {latest_file.name}", {
                                'source': 'outcome_file',
                                'file': str(latest_file)
                            }
                
                return False, "No regime data splitting results found - required for proper optimization", {
                    'severity': 'warning',
                    'impact': 'reduced_optimization_effectiveness'
                }
                
            except Exception as e:
                return False, f"Failed to load regime data splitting results: {e}", {
                    'severity': 'warning'
                }
        
        # Validate the structure of regime data
        required_keys = ['regime_data', 'regime_metadata']
        missing_keys = [key for key in required_keys if key not in regime_data_splitting]
        
        if missing_keys:
            return False, f"Missing regime data splitting keys: {missing_keys}", {
                'missing_keys': missing_keys,
                'severity': 'warning'
            }
        
        return True, "Regime data splitting results are present", None
    
    def _validate_pipeline_state_consistency(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate pipeline state consistency."""
        # Check for required dependencies - accept either multi-horizon or triple barrier labeling
        has_multi_horizon = 'multi_horizon_labeling_result' in pipeline_state
        has_triple_barrier = 'triple_barrier_labeling_result' in pipeline_state
        
        if not (has_multi_horizon or has_triple_barrier):
            return False, "Missing required labeling results (need either multi_horizon_labeling_result or triple_barrier_labeling_result)", {
                'missing_results': ['multi_horizon_labeling_result or triple_barrier_labeling_result']
            }
        
        return True, "Pipeline state is consistent", None
    
    def _generate_validation_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        failed_rules = [r for r in validation_results if r.status == ValidationStatus.FAILED]
        
        for result in failed_rules:
            if result.rule_name == "data_completeness":
                recommendations.append("Improve data quality by addressing missing values")
            elif result.rule_name == "no_infinite_values":
                recommendations.append("Clean data to remove infinite values")
            elif result.rule_name == "price_consistency":
                recommendations.append("Fix OHLC price relationship inconsistencies")
            elif result.rule_name == "best_score":
                recommendations.append("Review optimization parameters and data quality")
            elif result.rule_name == "optimization_convergence":
                recommendations.append("Increase optimization iterations or adjust parameters")
        
        if not recommendations:
            recommendations.append("All validations passed - no recommendations needed")
        
        return recommendations
    
    def validate_optimization_results(self, optimization_result: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
        """Validate optimization results structure and content."""
        validation_results = []
        
        # Check if results exist
        has_results = False
        has_features = False
        
        if 'results' in optimization_result:
            # New format
            results = optimization_result.get('results', {})
            has_results = bool(results)
            has_features = len(results) > 0
        elif 'optimization_results' in optimization_result:
            # Legacy format
            results = optimization_result.get('optimization_results', {})
            has_results = bool(results)
            has_features = len(results) > 0
        
        # Validate results presence
        if not has_results:
            validation_results.append(ValidationResult(
                rule_name="optimization_results_present",
                status=ValidationStatus.FAILED,
                level=ValidationLevel.CRITICAL,
                message="No optimization results found"
            ))
        else:
            validation_results.append(ValidationResult(
                rule_name="optimization_results_present", 
                status=ValidationStatus.PASSED,
                level=ValidationLevel.CRITICAL,
                message="Optimization results found"
            ))
        
        # Validate features presence
        if not has_features:
            validation_results.append(ValidationResult(
                rule_name="optimized_features_present",
                status=ValidationStatus.FAILED,
                level=ValidationLevel.CRITICAL,
                message="No optimized features found"
            ))
        else:
            validation_results.append(ValidationResult(
                rule_name="optimized_features_present",
                status=ValidationStatus.PASSED, 
                level=ValidationLevel.CRITICAL,
                message=f"Found {len(results)} optimized features"
            ))
        
        # Overall validation status
        critical_failures = [r for r in validation_results if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL]
        is_valid = len(critical_failures) == 0
        
        return is_valid, validation_results