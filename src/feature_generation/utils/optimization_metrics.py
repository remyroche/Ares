"""
import warnings
Optimization Performance Metrics and Reporting

This module provides comprehensive performance metrics and reporting capabilities
for feature lookback optimization results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
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

@dataclass
class OptimizationMetrics:
    """Comprehensive metrics for optimization results."""
    # Basic metrics
    total_features_optimized: int
    optimization_duration_seconds: float
    optimization_method: str
    
    # Performance metrics
    average_performance_score: float
    performance_score_std: float
    best_performing_feature: str
    worst_performing_feature: str
    
    # Stability metrics
    average_stability_score: float
    stability_score_std: float
    most_stable_feature: str
    least_stable_feature: str
    
    # Lookback distribution metrics
    average_lookback: float
    lookback_std: float
    min_lookback: int
    max_lookback: int
    lookback_diversity_score: float
    
    # Validation metrics
    validation_passed: bool
    validation_score: float
    total_warnings: int
    total_errors: int
    
    # Feature-specific metrics
    feature_metrics: Dict[str, Dict[str, Any]]
    
    # Timestamp
    timestamp: str

class OptimizationReporter:
    """
    Generates comprehensive reports for optimization results.
    
    This class provides detailed reporting capabilities for feature lookback
    optimization, including performance analysis, stability assessment, and
    actionable recommendations.
    """
    
    def __init__(self):
        """Initialize the optimization reporter."""
        self.logger = logger.getChild('OptimizationReporter')
        self.logger.info("Initializing OptimizationReporter")
    
    def generate_comprehensive_metrics(
        self, 
        optimization_results: Dict[str, Any],
        optimization_duration: float = 0.0
    ) -> OptimizationMetrics:
        """
        Generate comprehensive metrics from optimization results.
        
        Args:
            optimization_results: Results from optimization process
            optimization_duration: Duration of optimization in seconds
            
        Returns:
            OptimizationMetrics object with comprehensive metrics
        """
        self.logger.info("Generating comprehensive optimization metrics")
        
        try:
            # Extract basic information
            optimal_lookbacks = optimization_results.get('optimal_lookbacks', {})
            optimization_metrics = optimization_results.get('optimization_metrics', {})
            validation_result = optimization_results.get('validation_result', {})
            
            # Calculate basic metrics
            total_features = len(optimal_lookbacks)
            optimization_method = optimization_metrics.get('method', 'unknown')
            
            # Calculate performance metrics
            performance_scores = []
            stability_scores = []
            feature_metrics = {}
            
            for feature, lookback in optimal_lookbacks.items():
                # Create feature-specific metrics
                feature_metric = {
                    'optimal_lookback': lookback,
                    'performance_score': 0.0,  # Will be calculated if available
                    'stability_score': 0.0,    # Will be calculated if available
                    'validation_warnings': 0,
                    'validation_errors': 0
                }
                
                # Try to extract performance and stability scores
                if 'validation_result' in optimization_results:
                    validation_details = optimization_results['validation_result']
                    if 'performance_scores' in validation_details:
                        feature_metric['performance_score'] = validation_details['performance_scores'].get(feature, 0.0)
                        performance_scores.append(feature_metric['performance_score'])
                    
                    if 'stability_scores' in validation_details:
                        feature_metric['stability_score'] = validation_details['stability_scores'].get(feature, 0.0)
                        stability_scores.append(feature_metric['stability_score'])
                
                feature_metrics[feature] = feature_metric
            
            # Calculate aggregate metrics
            avg_performance = np.mean(performance_scores) if performance_scores else 0.0
            std_performance = np.std(performance_scores) if performance_scores else 0.0
            avg_stability = np.mean(stability_scores) if stability_scores else 0.0
            std_stability = np.std(stability_scores) if stability_scores else 0.0
            
            # Find best/worst performing features
            best_feature = max(optimal_lookbacks.keys(), 
                             key=lambda f: feature_metrics.get(f, {}).get('performance_score', 0)) if optimal_lookbacks else ""
            worst_feature = min(optimal_lookbacks.keys(), 
                              key=lambda f: feature_metrics.get(f, {}).get('performance_score', 0)) if optimal_lookbacks else ""
            
            # Find most/least stable features
            most_stable = max(optimal_lookbacks.keys(), 
                            key=lambda f: feature_metrics.get(f, {}).get('stability_score', 0)) if optimal_lookbacks else ""
            least_stable = min(optimal_lookbacks.keys(), 
                             key=lambda f: feature_metrics.get(f, {}).get('stability_score', 0)) if optimal_lookbacks else ""
            
            # Calculate lookback distribution metrics
            lookback_values = list(optimal_lookbacks.values())
            avg_lookback = np.mean(lookback_values) if lookback_values else 0.0
            std_lookback = np.std(lookback_values) if lookback_values else 0.0
            min_lookback = min(lookback_values) if lookback_values else 0
            max_lookback = max(lookback_values) if lookback_values else 0
            
            # Calculate lookback diversity score
            if len(lookback_values) > 1 and avg_lookback > 0:
                diversity_score = std_lookback / avg_lookback
            else:
                diversity_score = 0.0
            
            # Extract validation metrics
            validation_passed = validation_result.get('is_valid', True)
            validation_score = validation_result.get('overall_score', 1.0)
            total_warnings = len(validation_result.get('warnings', []))
            total_errors = len(validation_result.get('errors', []))
            
            metrics = OptimizationMetrics(
                total_features_optimized=total_features,
                optimization_duration_seconds=optimization_duration,
                optimization_method=optimization_method,
                average_performance_score=avg_performance,
                performance_score_std=std_performance,
                best_performing_feature=best_feature,
                worst_performing_feature=worst_feature,
                average_stability_score=avg_stability,
                stability_score_std=std_stability,
                most_stable_feature=most_stable,
                least_stable_feature=least_stable,
                average_lookback=avg_lookback,
                lookback_std=std_lookback,
                min_lookback=min_lookback,
                max_lookback=max_lookback,
                lookback_diversity_score=diversity_score,
                validation_passed=validation_passed,
                validation_score=validation_score,
                total_warnings=total_warnings,
                total_errors=total_errors,
                feature_metrics=feature_metrics,
                timestamp=datetime.now().isoformat()
            )
            
            self.logger.info(f"Generated metrics for {total_features} features")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive metrics: {e}")
            # Return minimal metrics
            return OptimizationMetrics(
                total_features_optimized=0,
                optimization_duration_seconds=optimization_duration,
                optimization_method='error',
                average_performance_score=0.0,
                performance_score_std=0.0,
                best_performing_feature='',
                worst_performing_feature='',
                average_stability_score=0.0,
                stability_score_std=0.0,
                most_stable_feature='',
                least_stable_feature='',
                average_lookback=0.0,
                lookback_std=0.0,
                min_lookback=0,
                max_lookback=0,
                lookback_diversity_score=0.0,
                validation_passed=False,
                validation_score=0.0,
                total_warnings=0,
                total_errors=1,
                feature_metrics={},
                timestamp=datetime.now().isoformat()
            )
    
    def generate_performance_report(self, metrics: OptimizationMetrics) -> str:
        """Generate a detailed performance report."""
        report = []
        report.append("=" * 80)
        report.append("FEATURE LOOKBACK OPTIMIZATION PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {metrics.timestamp}")
        report.append(f"Optimization Method: {metrics.optimization_method}")
        report.append(f"Duration: {metrics.optimization_duration_seconds:.2f} seconds")
        report.append("")
        
        # Summary section
        report.append("📊 SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Features Optimized: {metrics.total_features_optimized}")
        report.append(f"Validation Status: {'✅ PASSED' if metrics.validation_passed else '❌ FAILED'}")
        report.append(f"Validation Score: {metrics.validation_score:.3f}")
        report.append(f"Total Warnings: {metrics.total_warnings}")
        report.append(f"Total Errors: {metrics.total_errors}")
        report.append("")
        
        # Performance section
        report.append("🎯 PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append(f"Average Performance Score: {metrics.average_performance_score:.3f} ± {metrics.performance_score_std:.3f}")
        report.append(f"Best Performing Feature: {metrics.best_performing_feature}")
        report.append(f"Worst Performing Feature: {metrics.worst_performing_feature}")
        report.append("")
        
        # Stability section
        report.append("🔒 STABILITY METRICS")
        report.append("-" * 40)
        report.append(f"Average Stability Score: {metrics.average_stability_score:.3f} ± {metrics.stability_score_std:.3f}")
        report.append(f"Most Stable Feature: {metrics.most_stable_feature}")
        report.append(f"Least Stable Feature: {metrics.least_stable_feature}")
        report.append("")
        
        # Lookback distribution section
        report.append("📈 LOOKBACK DISTRIBUTION")
        report.append("-" * 40)
        report.append(f"Average Lookback: {metrics.average_lookback:.1f} ± {metrics.lookback_std:.1f}")
        report.append(f"Lookback Range: {metrics.min_lookback} - {metrics.max_lookback}")
        report.append(f"Diversity Score: {metrics.lookback_diversity_score:.3f}")
        report.append("")
        
        # Feature details section
        if metrics.feature_metrics:
            report.append("🔍 FEATURE DETAILS")
            report.append("-" * 40)
            for feature, feature_metric in metrics.feature_metrics.items():
                report.append(f"\n{feature.upper()}:")
                report.append(f"  Optimal Lookback: {feature_metric.get('optimal_lookback', 'N/A')}")
                report.append(f"  Performance Score: {feature_metric.get('performance_score', 0.0):.3f}")
                report.append(f"  Stability Score: {feature_metric.get('stability_score', 0.0):.3f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def generate_recommendations(self, metrics: OptimizationMetrics) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        
        # Performance-based recommendations
        if metrics.average_performance_score < 0.3:
            recommendations.append("Consider expanding the optimization period range or using different optimization methods")
        
        if metrics.performance_score_std > 0.5:
            recommendations.append("High variance in performance scores suggests inconsistent optimization - review feature selection")
        
        # Stability-based recommendations
        if metrics.average_stability_score < 0.5:
            recommendations.append("Low stability scores indicate unreliable optimization - increase data size or use more robust methods")
        
        if metrics.stability_score_std > 0.3:
            recommendations.append("High variance in stability suggests some features are more reliable than others - focus on stable features")
        
        # Lookback distribution recommendations
        if metrics.lookback_diversity_score < 0.1:
            recommendations.append("Very low lookback diversity suggests similar optimal periods - consider expanding period ranges")
        
        if metrics.max_lookback > 100:
            recommendations.append("Very long lookback periods detected - verify data quality and consider shorter periods")
        
        if metrics.min_lookback < 3:
            recommendations.append("Very short lookback periods detected - may indicate overfitting or insufficient data")
        
        # Validation-based recommendations
        if not metrics.validation_passed:
            recommendations.append("Validation failed - review optimization implementation and data quality")
        
        if metrics.total_warnings > 5:
            recommendations.append("High number of warnings - review optimization parameters and data preprocessing")
        
        if metrics.total_errors > 0:
            recommendations.append("Errors detected in optimization - fix implementation issues before proceeding")
        
        # General recommendations
        if metrics.optimization_duration_seconds > 300:  # 5 minutes
            recommendations.append("Long optimization duration - consider parallel processing or reducing parameter space")
        
        if metrics.total_features_optimized < 3:
            recommendations.append("Few features optimized - consider adding more features for comprehensive analysis")
        
        return recommendations
    
    def export_metrics_to_json(self, metrics: OptimizationMetrics, filepath: str) -> bool:
        """Export metrics to JSON file."""
        try:
            # Convert dataclass to dictionary
            metrics_dict = asdict(metrics)
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            self.logger.info(f"Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics to JSON: {e}")
            return False
    
    def export_metrics_to_csv(self, metrics: OptimizationMetrics, filepath: str) -> bool:
        """Export feature metrics to CSV file."""
        try:
            if not metrics.feature_metrics:
                self.logger.warning("No feature metrics to export")
                return False
            
            # Create DataFrame from feature metrics
            df_data = []
            for feature, feature_metric in metrics.feature_metrics.items():
                row = {
                    'feature': feature,
                    'optimal_lookback': feature_metric.get('optimal_lookback', 0),
                    'performance_score': feature_metric.get('performance_score', 0.0),
                    'stability_score': feature_metric.get('stability_score', 0.0),
                    'validation_warnings': feature_metric.get('validation_warnings', 0),
                    'validation_errors': feature_metric.get('validation_errors', 0)
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Feature metrics exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics to CSV: {e}")
            return False

# Convenience functions
def generate_optimization_report(
    optimization_results: Dict[str, Any],
    optimization_duration: float = 0.0
) -> Tuple[OptimizationMetrics, str, List[str]]:
    """
    Generate comprehensive optimization report.
    
    Returns:
        Tuple of (metrics, report_text, recommendations)
    """
    reporter = OptimizationReporter()
    metrics = reporter.generate_comprehensive_metrics(optimization_results, optimization_duration)
    report = reporter.generate_performance_report(metrics)
    recommendations = reporter.generate_recommendations(metrics)
    
    return metrics, report, recommendations

def quick_metrics_summary(optimization_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate quick metrics summary."""
    reporter = OptimizationReporter()
    metrics = reporter.generate_comprehensive_metrics(optimization_results)
    
    return {
        'total_features': metrics.total_features_optimized,
        'validation_passed': metrics.validation_passed,
        'validation_score': metrics.validation_score,
        'average_performance': metrics.average_performance_score,
        'average_stability': metrics.average_stability_score,
        'average_lookback': metrics.average_lookback,
        'total_warnings': metrics.total_warnings,
        'total_errors': metrics.total_errors
    }
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
