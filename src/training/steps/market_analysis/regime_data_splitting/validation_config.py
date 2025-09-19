"""
Unified validation configuration for regime data splitting module.

This module provides consistent validation thresholds, rules, and configuration
across all components in the regime data splitting module.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class ValidationSeverity(Enum):
    """Validation severity levels."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class DataValidationThresholds:
    """Data validation thresholds."""
    # Data quality thresholds
    min_data_quality_score: float = 0.7
    max_missing_data_percentage: float = 10.0
    max_duplicate_data_percentage: float = 5.0
    max_infinite_values_percentage: float = 1.0
    max_invalid_prices_percentage: float = 2.0
    
    # Data size thresholds
    min_data_rows: int = 100
    max_data_rows: int = 1_000_000
    min_data_columns: int = 4  # OHLC minimum
    max_data_columns: int = 1000
    
    # Data alignment thresholds
    max_acceptable_data_loss_percentage: float = 5.0
    critical_data_loss_percentage: float = 20.0
    min_alignment_length: int = 50
    
    # Temporal consistency thresholds
    require_monotonic_timestamps: bool = True
    max_timestamp_gaps_percentage: float = 5.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


@dataclass
class RegimeValidationThresholds:
    """Regime validation thresholds."""
    # Regime count thresholds
    min_regimes: int = 2
    max_regimes: int = 20
    optimal_min_regimes: int = 3
    optimal_max_regimes: int = 10
    
    # Regime distribution thresholds
    min_regime_points_percentage: float = 1.0  # Minimum 1% of data per regime
    max_regime_points_percentage: float = 90.0  # Maximum 90% of data in one regime
    
    # Regime continuity thresholds
    min_continuity_score: float = 0.5
    optimal_continuity_score: float = 0.7
    max_transition_rate: float = 0.3  # Max 30% of points can be transitions
    
    # Regime confidence thresholds
    min_confidence_score: float = 0.6
    optimal_confidence_score: float = 0.8
    min_mean_confidence: float = 0.7
    
    # Regime validation ranges
    min_regime_value: int = 0
    max_regime_value: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


@dataclass
class PerformanceValidationThresholds:
    """Performance validation thresholds."""
    # Execution time thresholds (seconds)
    max_execution_time: float = 300.0  # 5 minutes
    warning_execution_time: float = 60.0  # 1 minute
    
    # Memory usage thresholds (MB)
    max_memory_usage: float = 2000.0  # 2GB
    warning_memory_usage: float = 1000.0  # 1GB
    
    # Processing rate thresholds (rows per second)
    min_processing_rate: float = 100.0
    optimal_processing_rate: float = 1000.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


@dataclass
class ValidationConfiguration:
    """Unified validation configuration."""
    data_thresholds: DataValidationThresholds = field(default_factory=DataValidationThresholds)
    regime_thresholds: RegimeValidationThresholds = field(default_factory=RegimeValidationThresholds)
    performance_thresholds: PerformanceValidationThresholds = field(default_factory=PerformanceValidationThresholds)
    
    # Global validation settings
    enable_strict_validation: bool = True
    enable_performance_validation: bool = True
    enable_temporal_validation: bool = True
    enable_regime_quality_validation: bool = True
    
    # Validation behavior
    fail_on_warnings: bool = False
    collect_detailed_metrics: bool = True
    log_validation_details: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'data_thresholds': self.data_thresholds.to_dict(),
            'regime_thresholds': self.regime_thresholds.to_dict(),
            'performance_thresholds': self.performance_thresholds.to_dict(),
            'enable_strict_validation': self.enable_strict_validation,
            'enable_performance_validation': self.enable_performance_validation,
            'enable_temporal_validation': self.enable_temporal_validation,
            'enable_regime_quality_validation': self.enable_regime_quality_validation,
            'fail_on_warnings': self.fail_on_warnings,
            'collect_detailed_metrics': self.collect_detailed_metrics,
            'log_validation_details': self.log_validation_details
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ValidationConfiguration':
        """Create from dictionary."""
        data_thresholds = DataValidationThresholds(**config_dict.get('data_thresholds', {}))
        regime_thresholds = RegimeValidationThresholds(**config_dict.get('regime_thresholds', {}))
        performance_thresholds = PerformanceValidationThresholds(**config_dict.get('performance_thresholds', {}))
        
        return cls(
            data_thresholds=data_thresholds,
            regime_thresholds=regime_thresholds,
            performance_thresholds=performance_thresholds,
            enable_strict_validation=config_dict.get('enable_strict_validation', True),
            enable_performance_validation=config_dict.get('enable_performance_validation', True),
            enable_temporal_validation=config_dict.get('enable_temporal_validation', True),
            enable_regime_quality_validation=config_dict.get('enable_regime_quality_validation', True),
            fail_on_warnings=config_dict.get('fail_on_warnings', False),
            collect_detailed_metrics=config_dict.get('collect_detailed_metrics', True),
            log_validation_details=config_dict.get('log_validation_details', True)
        )


# Validation rule definitions
@dataclass
class ValidationRule:
    """Definition of a validation rule."""
    name: str
    description: str
    threshold_key: str
    comparison: str  # 'gt', 'lt', 'gte', 'lte', 'eq', 'ne', 'range'
    severity: ValidationSeverity
    error_message: str
    action_required: str
    threshold_value: Optional[Any] = None
    threshold_range: Optional[tuple] = None


# Predefined validation rules
DATA_VALIDATION_RULES = [
    ValidationRule(
        name="min_data_rows",
        description="Minimum number of data rows required",
        threshold_key="min_data_rows",
        comparison="gte",
        severity=ValidationSeverity.ERROR,
        error_message="Insufficient data rows",
        action_required="Provide dataset with more rows"
    ),
    ValidationRule(
        name="data_quality_score",
        description="Overall data quality score",
        threshold_key="min_data_quality_score",
        comparison="gte",
        severity=ValidationSeverity.WARNING,
        error_message="Data quality score below threshold",
        action_required="Improve data quality by addressing missing values, duplicates, and invalid data"
    ),
    ValidationRule(
        name="missing_data_percentage",
        description="Percentage of missing data",
        threshold_key="max_missing_data_percentage",
        comparison="lte",
        severity=ValidationSeverity.WARNING,
        error_message="High percentage of missing data",
        action_required="Fill or remove missing values"
    ),
    ValidationRule(
        name="data_alignment_loss",
        description="Data loss during alignment",
        threshold_key="max_acceptable_data_loss_percentage",
        comparison="lte",
        severity=ValidationSeverity.WARNING,
        error_message="Significant data loss during alignment",
        action_required="Review data alignment logic and time ranges"
    ),
    ValidationRule(
        name="critical_data_alignment_loss",
        description="Critical data loss during alignment",
        threshold_key="critical_data_loss_percentage",
        comparison="lt",
        severity=ValidationSeverity.CRITICAL,
        error_message="Critical data loss during alignment",
        action_required="Fix data alignment issues or provide compatible time ranges"
    )
]

REGIME_VALIDATION_RULES = [
    ValidationRule(
        name="min_regime_count",
        description="Minimum number of regimes",
        threshold_key="min_regimes",
        comparison="gte",
        severity=ValidationSeverity.ERROR,
        error_message="Too few regimes detected",
        action_required="Adjust regime discovery parameters or provide more diverse data"
    ),
    ValidationRule(
        name="max_regime_count",
        description="Maximum number of regimes",
        threshold_key="max_regimes",
        comparison="lte",
        severity=ValidationSeverity.WARNING,
        error_message="Too many regimes detected",
        action_required="Consider reducing regime complexity or adjusting parameters"
    ),
    ValidationRule(
        name="regime_continuity_score",
        description="Regime continuity score",
        threshold_key="min_continuity_score",
        comparison="gte",
        severity=ValidationSeverity.WARNING,
        error_message="Low regime continuity score",
        action_required="Consider smoothing parameters to reduce regime transitions"
    ),
    ValidationRule(
        name="regime_confidence_score",
        description="Average regime confidence score",
        threshold_key="min_mean_confidence",
        comparison="gte",
        severity=ValidationSeverity.WARNING,
        error_message="Low regime confidence score",
        action_required="Review regime discovery quality or model parameters"
    )
]

PERFORMANCE_VALIDATION_RULES = [
    ValidationRule(
        name="execution_time",
        description="Maximum execution time",
        threshold_key="max_execution_time",
        comparison="lte",
        severity=ValidationSeverity.WARNING,
        error_message="Execution time exceeded threshold",
        action_required="Optimize processing or reduce data size"
    ),
    ValidationRule(
        name="memory_usage",
        description="Maximum memory usage",
        threshold_key="max_memory_usage",
        comparison="lte",
        severity=ValidationSeverity.WARNING,
        error_message="Memory usage exceeded threshold",
        action_required="Optimize memory usage or use streaming processing"
    )
]


class UnifiedValidator:
    """Unified validator using consistent thresholds and rules."""
    
    def __init__(self, config: Optional[ValidationConfiguration] = None):
        self.config = config or ValidationConfiguration()
        self.validation_rules = {
            'data': DATA_VALIDATION_RULES,
            'regime': REGIME_VALIDATION_RULES,
            'performance': PERFORMANCE_VALIDATION_RULES
        }
    
    def validate_data_quality(self, data_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality using unified thresholds."""
        return self._apply_validation_rules('data', data_metrics, self.config.data_thresholds.to_dict())
    
    def validate_regime_quality(self, regime_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate regime quality using unified thresholds."""
        return self._apply_validation_rules('regime', regime_metrics, self.config.regime_thresholds.to_dict())
    
    def validate_performance(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance using unified thresholds."""
        return self._apply_validation_rules('performance', performance_metrics, self.config.performance_thresholds.to_dict())
    
    def _apply_validation_rules(self, rule_category: str, metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Apply validation rules to metrics."""
        validation_result = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'info': [],
            'details': {}
        }
        
        rules = self.validation_rules.get(rule_category, [])
        
        for rule in rules:
            try:
                result = self._evaluate_rule(rule, metrics, thresholds)
                validation_result['details'][rule.name] = result
                
                if not result['passed']:
                    message = f"{rule.error_message}. Action required: {rule.action_required}"
                    
                    if rule.severity == ValidationSeverity.CRITICAL:
                        validation_result['errors'].append(message)
                        validation_result['passed'] = False
                    elif rule.severity == ValidationSeverity.ERROR:
                        validation_result['errors'].append(message)
                        validation_result['passed'] = False
                    elif rule.severity == ValidationSeverity.WARNING:
                        validation_result['warnings'].append(message)
                        if self.config.fail_on_warnings:
                            validation_result['passed'] = False
                    else:
                        validation_result['info'].append(message)
                        
            except Exception as e:
                error_msg = f"Validation rule {rule.name} failed: {str(e)}"
                validation_result['errors'].append(error_msg)
                validation_result['passed'] = False
        
        return validation_result
    
    def _evaluate_rule(self, rule: ValidationRule, metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single validation rule."""
        # Get the metric value
        metric_keys = rule.threshold_key.replace('min_', '').replace('max_', '')
        metric_value = metrics.get(metric_keys, metrics.get(rule.threshold_key))
        
        if metric_value is None:
            return {
                'passed': False,
                'reason': f"Metric {rule.threshold_key} not found in metrics",
                'metric_value': None,
                'threshold_value': None
            }
        
        # Get the threshold value
        threshold_value = thresholds.get(rule.threshold_key)
        if threshold_value is None:
            return {
                'passed': False,
                'reason': f"Threshold {rule.threshold_key} not found",
                'metric_value': metric_value,
                'threshold_value': None
            }
        
        # Evaluate the comparison
        passed = self._compare_values(metric_value, threshold_value, rule.comparison)
        
        return {
            'passed': passed,
            'metric_value': metric_value,
            'threshold_value': threshold_value,
            'comparison': rule.comparison,
            'reason': f"{metric_value} {rule.comparison} {threshold_value}" if not passed else "Passed"
        }
    
    def _compare_values(self, value: Any, threshold: Any, comparison: str) -> bool:
        """Compare values according to comparison operator."""
        try:
            if comparison == 'gt':
                return value > threshold
            elif comparison == 'lt':
                return value < threshold
            elif comparison == 'gte':
                return value >= threshold
            elif comparison == 'lte':
                return value <= threshold
            elif comparison == 'eq':
                return value == threshold
            elif comparison == 'ne':
                return value != threshold
            elif comparison == 'range':
                if isinstance(threshold, (list, tuple)) and len(threshold) == 2:
                    return threshold[0] <= value <= threshold[1]
                return False
            else:
                return False
        except Exception:
            return False


# Global validator instance
_global_validator = None

def get_unified_validator(config: Optional[ValidationConfiguration] = None) -> UnifiedValidator:
    """Get or create a unified validator instance."""
    global _global_validator
    
    if _global_validator is None:
        _global_validator = UnifiedValidator(config)
    
    return _global_validator

def reset_unified_validator():
    """Reset the global unified validator (useful for testing)."""
    global _global_validator
    _global_validator = None