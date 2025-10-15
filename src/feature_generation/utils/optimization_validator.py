"""
import warnings
Optimization Validation Framework

This module provides comprehensive validation for feature lookback optimization results,
including statistical validation, stability checks, and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation levels for optimization results."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

@dataclass
class ValidationResult:
    """Result of optimization validation."""
    is_valid: bool
    validation_level: ValidationLevel
    overall_score: float
    validation_details: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]

class OptimizationValidator:
    """
    Validates feature lookback optimization results.
    
    This class provides comprehensive validation for optimization results,
    including statistical significance, stability, and performance metrics.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """Initialize the optimization validator."""
        self.logger = logger.getChild('OptimizationValidator')
        self.validation_level = validation_level
        self.logger.info(f"Initializing OptimizationValidator with level: {validation_level.value}")
    
    def validate_optimization_results(
        self, 
        optimization_results: Dict[str, Any],
        data: Optional[pd.DataFrame] = None,
        feature_generators: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate optimization results comprehensively.
        
        Args:
            optimization_results: Results from optimization process
            data: Original data used for optimization
            feature_generators: Feature generator functions
            
        Returns:
            ValidationResult with validation details
        """
        self.logger.info("Starting comprehensive validation of optimization results")
        
        validation_details = {}
        warnings = []
        errors = []
        recommendations = []
        
        try:
            # Basic validation
            basic_validation = self._validate_basic_structure(optimization_results)
            validation_details['basic_validation'] = basic_validation
            if not basic_validation['is_valid']:
                errors.extend(basic_validation['errors'])
            
            # Statistical validation
            if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                statistical_validation = self._validate_statistical_properties(optimization_results)
                validation_details['statistical_validation'] = statistical_validation
                warnings.extend(statistical_validation['warnings'])
                recommendations.extend(statistical_validation['recommendations'])
            
            # Stability validation
            if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                stability_validation = self._validate_stability(optimization_results)
                validation_details['stability_validation'] = stability_validation
                warnings.extend(stability_validation['warnings'])
                recommendations.extend(stability_validation['recommendations'])
            
            # Performance validation
            if self.validation_level == ValidationLevel.COMPREHENSIVE and data is not None:
                performance_validation = self._validate_performance(
                    optimization_results, data, feature_generators
                )
                validation_details['performance_validation'] = performance_validation
                warnings.extend(performance_validation['warnings'])
                recommendations.extend(performance_validation['recommendations'])
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(validation_details)
            
            # Determine if results are valid
            is_valid = len(errors) == 0 and overall_score >= 0.6
            
            result = ValidationResult(
                is_valid=is_valid,
                validation_level=self.validation_level,
                overall_score=overall_score,
                validation_details=validation_details,
                warnings=warnings,
                errors=errors,
                recommendations=recommendations
            )
            
            self.logger.info(f"Validation completed. Overall score: {overall_score:.3f}, Valid: {is_valid}")
            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed with error: {e}")
            return ValidationResult(
                is_valid=False,
                validation_level=self.validation_level,
                overall_score=0.0,
                validation_details={'error': str(e)},
                warnings=[],
                errors=[f"Validation failed: {e}"],
                recommendations=["Review optimization implementation"]
            )
    
    def _validate_basic_structure(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate basic structure of optimization results."""
        self.logger.debug("Validating basic structure")
        
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'score': 1.0
        }
        
        # Check required keys
        required_keys = ['optimal_lookbacks', 'optimization_metrics']
        for key in required_keys:
            if key not in optimization_results:
                validation['errors'].append(f"Missing required key: {key}")
                validation['is_valid'] = False
                validation['score'] = 0.0
        
        # Validate optimal_lookbacks
        if 'optimal_lookbacks' in optimization_results:
            optimal_lookbacks = optimization_results['optimal_lookbacks']
            if not isinstance(optimal_lookbacks, dict):
                validation['errors'].append("optimal_lookbacks must be a dictionary")
                validation['is_valid'] = False
                validation['score'] = 0.0
            else:
                # Check for reasonable lookback values
                for feature, lookback in optimal_lookbacks.items():
                    if not isinstance(lookback, (int, float)) or lookback <= 0:
                        validation['errors'].append(f"Invalid lookback value for {feature}: {lookback}")
                        validation['is_valid'] = False
                        validation['score'] = 0.0
                    elif lookback > 1000:  # Unreasonably large lookback
                        validation['warnings'].append(f"Very large lookback for {feature}: {lookback}")
        
        # Validate optimization_metrics
        if 'optimization_metrics' in optimization_results:
            metrics = optimization_results['optimization_metrics']
            if not isinstance(metrics, dict):
                validation['warnings'].append("optimization_metrics should be a dictionary")
        
        return validation
    
    def _validate_statistical_properties(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statistical properties of optimization results."""
        self.logger.debug("Validating statistical properties")
        
        validation = {
            'warnings': [],
            'recommendations': [],
            'score': 1.0
        }
        
        if 'optimal_lookbacks' not in optimization_results:
            return validation
        
        optimal_lookbacks = optimization_results['optimal_lookbacks']
        
        # Check for diversity in lookback periods
        lookback_values = list(optimal_lookbacks.values())
        if len(lookback_values) > 1:
            lookback_std = np.std(lookback_values)
            lookback_mean = np.mean(lookback_values)
            
            if lookback_std < lookback_mean * 0.1:  # Very low diversity
                validation['warnings'].append("Low diversity in optimal lookback periods")
                validation['recommendations'].append("Consider expanding the period range for optimization")
                validation['score'] *= 0.8
            
            # Check for extreme values
            for feature, lookback in optimal_lookbacks.items():
                if lookback < 2:
                    validation['warnings'].append(f"Very short lookback for {feature}: {lookback}")
                    validation['score'] *= 0.9
                elif lookback > 200:
                    validation['warnings'].append(f"Very long lookback for {feature}: {lookback}")
                    validation['score'] *= 0.9
        
        return validation
    
    def _validate_stability(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate stability of optimization results."""
        self.logger.debug("Validating stability")
        
        validation = {
            'warnings': [],
            'recommendations': [],
            'score': 1.0
        }
        
        # Check if stability information is available
        if 'optimization_metrics' in optimization_results:
            metrics = optimization_results['optimization_metrics']
            
            # Look for stability indicators
            if 'stability_scores' in metrics:
                stability_scores = metrics['stability_scores']
                if isinstance(stability_scores, dict):
                    low_stability_features = [
                        feature for feature, score in stability_scores.items()
                        if score < 0.5
                    ]
                    if low_stability_features:
                        validation['warnings'].append(
                            f"Low stability for features: {low_stability_features}"
                        )
                        validation['recommendations'].append(
                            "Consider using more data or different optimization methods"
                        )
                        validation['score'] *= 0.7
            
            # Check for confidence intervals
            if 'confidence_intervals' in metrics:
                confidence_intervals = metrics['confidence_intervals']
                if isinstance(confidence_intervals, dict):
                    wide_intervals = []
                    for feature, interval in confidence_intervals.items():
                        if isinstance(interval, (list, tuple)) and len(interval) == 2:
                            width = interval[1] - interval[0]
                            if width > 0.5:  # Wide confidence interval
                                wide_intervals.append(feature)
                    
                    if wide_intervals:
                        validation['warnings'].append(
                            f"Wide confidence intervals for features: {wide_intervals}"
                        )
                        validation['score'] *= 0.8
        
        return validation
    
    def _validate_performance(
        self, 
        optimization_results: Dict[str, Any], 
        data: pd.DataFrame,
        feature_generators: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate performance of optimization results."""
        self.logger.debug("Validating performance")
        
        validation = {
            'warnings': [],
            'recommendations': [],
            'score': 1.0
        }
        
        if 'optimal_lookbacks' not in optimization_results or feature_generators is None:
            return validation
        
        optimal_lookbacks = optimization_results['optimal_lookbacks']
        
        # Test performance of optimized features
        performance_scores = {}
        
        for feature, lookback in optimal_lookbacks.items():
            if feature in feature_generators:
                try:
                    generator = feature_generators[feature]
                    feature_values = generator(data, lookback)
                    
                    # Calculate performance metrics
                    if 'close' in data.columns:
                        # Calculate correlation with price changes
                        price_changes = data['close'].pct_change()
                        correlation = abs(feature_values.corr(price_changes))
                        performance_scores[feature] = correlation if not pd.isna(correlation) else 0
                    else:
                        # Use autocorrelation as fallback
                        autocorr = feature_values.autocorr(lag=1)
                        performance_scores[feature] = abs(autocorr) if not pd.isna(autocorr) else 0
                        
                except Exception as e:
                    self.logger.warning(f"Error validating performance for {feature}: {e}")
                    performance_scores[feature] = 0
        
        # Check for low performance features
        low_performance_features = [
            feature for feature, score in performance_scores.items()
            if score < 0.1
        ]
        
        if low_performance_features:
            validation['warnings'].append(
                f"Low performance features: {low_performance_features}"
            )
            validation['recommendations'].append(
                "Consider removing or redesigning low-performance features"
            )
            validation['score'] *= 0.6
        
        # Check for very high performance (potential overfitting)
        high_performance_features = [
            feature for feature, score in performance_scores.items()
            if score > 0.9
        ]
        
        if high_performance_features:
            validation['warnings'].append(
                f"Very high performance features (potential overfitting): {high_performance_features}"
            )
            validation['recommendations'].append(
                "Validate results on out-of-sample data"
            )
            validation['score'] *= 0.9
        
        validation['performance_scores'] = performance_scores
        return validation
    
    def _calculate_overall_score(self, validation_details: Dict[str, Any]) -> float:
        """Calculate overall validation score."""
        scores = []
        
        # Basic validation score
        if 'basic_validation' in validation_details:
            scores.append(validation_details['basic_validation'].get('score', 0.0))
        
        # Statistical validation score
        if 'statistical_validation' in validation_details:
            scores.append(validation_details['statistical_validation'].get('score', 1.0))
        
        # Stability validation score
        if 'stability_validation' in validation_details:
            scores.append(validation_details['stability_validation'].get('score', 1.0))
        
        # Performance validation score
        if 'performance_validation' in validation_details:
            scores.append(validation_details['performance_validation'].get('score', 1.0))
        
        if not scores:
            return 0.0
        
        # Weighted average (basic validation is most important)
        weights = [0.4, 0.2, 0.2, 0.2][:len(scores)]
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return min(1.0, max(0.0, weighted_score))
    
    def generate_validation_report(self, validation_result: ValidationResult) -> str:
        """Generate a human-readable validation report."""
        report = []
        report.append("=" * 60)
        report.append("FEATURE LOOKBACK OPTIMIZATION VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Validation Level: {validation_result.validation_level.value}")
        report.append(f"Overall Score: {validation_result.overall_score:.3f}")
        report.append(f"Valid: {'✅ YES' if validation_result.is_valid else '❌ NO'}")
        report.append("")
        
        if validation_result.errors:
            report.append("🚨 ERRORS:")
            for error in validation_result.errors:
                report.append(f"  • {error}")
            report.append("")
        
        if validation_result.warnings:
            report.append("⚠️ WARNINGS:")
            for warning in validation_result.warnings:
                report.append(f"  • {warning}")
            report.append("")
        
        if validation_result.recommendations:
            report.append("💡 RECOMMENDATIONS:")
            for recommendation in validation_result.recommendations:
                report.append(f"  • {recommendation}")
            report.append("")
        
        # Add detailed validation results
        report.append("📊 DETAILED VALIDATION RESULTS:")
        for section, details in validation_result.validation_details.items():
            report.append(f"\n{section.upper().replace('_', ' ')}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    if key not in ['warnings', 'errors', 'recommendations']:
                        report.append(f"  {key}: {value}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

# Convenience functions
def validate_optimization_results(
    optimization_results: Dict[str, Any],
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    data: Optional[pd.DataFrame] = None,
    feature_generators: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """Convenience function for validating optimization results."""
    validator = OptimizationValidator(validation_level)
    return validator.validate_optimization_results(
        optimization_results, data, feature_generators
    )

def quick_validate(optimization_results: Dict[str, Any]) -> bool:
    """Quick validation check - returns True if results are valid."""
    validator = OptimizationValidator(ValidationLevel.BASIC)
    result = validator.validate_optimization_results(optimization_results)
    return result.is_valid
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
