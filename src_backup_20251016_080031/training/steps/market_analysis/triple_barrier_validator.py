"""
MARKET_ANALYSIS Triple Barrier Validation Framework

This module provides comprehensive validation for triple barrier labeling implementations.
It includes data quality validation, labeling accuracy validation, and performance validation.

Key Features:
- Data quality validation
- Labeling accuracy validation
- Performance metrics validation
- Temporal validation (lookahead bias detection)
- Statistical validation
- Integration with market analysis pipeline
"""

from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time
from src.utils.math_validation import safe_divide, validate_positive, MathValidationError

import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import contextlib
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings

@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    score: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = 'info'  # info, warning, error, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'passed': self.passed,
            'score': self.score,
            'message': self.message,
            'details': self.details,
            'severity': self.severity
        }

@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    overall_score: float
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    critical_issues: List[str]
    validation_results: Dict[str, ValidationResult]
    recommendations: List[str]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall_score': self.overall_score,
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'warning_checks': self.warning_checks,
            'critical_issues': self.critical_issues,
            'validation_results': {k: v.to_dict() for k, v in self.validation_results.items()},
            'recommendations': self.recommendations,
            'timestamp': self.timestamp
        }

class TripleBarrierValidator:
    """
    Comprehensive validator for triple barrier labeling implementations.
    
    This class provides validation for data quality, labeling accuracy,
    performance metrics, and temporal consistency.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the triple barrier validator.
        
        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config or {}
        self.logger = get_logger('TripleBarrierValidator')
        
        # Default validation parameters
        self.validation_params = {
            'min_data_points': 100,
            'max_missing_ratio': 0.05,
            'min_price_change': 0.0001,
            'max_price_change': 0.5,
            'min_label_ratio': 0.01,
            'max_imbalance_ratio': 10.0,
            'min_win_rate': 0.3,
            'max_drawdown_threshold': 0.2,
            'min_sharpe_ratio': 0.5,
            'temporal_validation': True,
            'statistical_validation': True,
            'performance_validation': True
        }
        
        # Update with provided config
        self.validation_params.update(self.config)
        
        self._log_initialization()
    
    def _log_initialization(self):
        """Log initialization parameters."""
        self.logger.info('🚀 Initializing Triple Barrier Validator')
        self.logger.info(f'📋 Validation parameters:')
        self.logger.info(f'   → Min data points: {self.validation_params["min_data_points"]}')
        self.logger.info(f'   → Max missing ratio: {self.validation_params["max_missing_ratio"]}')
        self.logger.info(f'   → Min label ratio: {self.validation_params["min_label_ratio"]}')
        self.logger.info(f'   → Max imbalance ratio: {self.validation_params["max_imbalance_ratio"]}')
        self.logger.info(f'   → Min win rate: {self.validation_params["min_win_rate"]}')
    
    @traced(span_name='validate_triple_barrier_implementation')
    @validates()
    @handles_errors(exceptions=(Exception,), default_return=ValidationReport(0.0, 0, 0, 0, 0, [], {}, [], datetime.now().isoformat()))
    @log_execution_time()
    def validate_triple_barrier_implementation(
        self, 
        data: pd.DataFrame, 
        labeled_data: Optional[pd.DataFrame] = None
    ) -> ValidationReport:
        """Validate a triple barrier labeling implementation.
        
        Args:
            data: Original input data
            labeled_data: Data with triple barrier labels (optional)
            
        Returns:
            Comprehensive validation report
        """
        self.logger.info('🔍 Starting comprehensive triple barrier validation')
        self.logger.info(f'   Input data shape: {data.shape}')
        if labeled_data is not None:
            self.logger.info(f'   Labeled data shape: {labeled_data.shape}')
        
        validation_results = {}
        
        # Data quality validation
        data_quality_result = self._validate_data_quality(data)
        validation_results['data_quality'] = data_quality_result
        
        if labeled_data is not None:
            # Labeling validation
            labeling_result = self._validate_labeling_quality(data, labeled_data)
            validation_results['labeling_quality'] = labeling_result
            
            # Performance validation
            performance_result = self._validate_performance(labeled_data)
            validation_results['performance'] = performance_result
            
            # Temporal validation
            if self.validation_params['temporal_validation']:
                temporal_result = self._validate_temporal_consistency(data, labeled_data)
                validation_results['temporal_consistency'] = temporal_result
            
            # Statistical validation
            if self.validation_params['statistical_validation']:
                statistical_result = self._validate_statistical_properties(labeled_data)
                validation_results['statistical_properties'] = statistical_result
        
        # Generate comprehensive report
        report = self._generate_validation_report(validation_results)
        
        self.logger.info(f'✅ Validation completed - Overall score: {report.overall_score:.3f}')
        self.logger.info(f'   → Passed: {report.passed_checks}/{report.total_checks}')
        self.logger.info(f'   → Warnings: {report.warning_checks}')
        self.logger.info(f'   → Critical issues: {len(report.critical_issues)}')
        
        return report
    
    def _validate_data_quality(self, data: pd.DataFrame) -> ValidationResult:
        """Validate input data quality."""
        self.logger.info('📊 Validating data quality...')
        
        issues = []
        score = 1.0
        
        # Check data size
        if len(data) < self.validation_params['min_data_points']:
            issues.append(f'Insufficient data points: {len(data)} < {self.validation_params["min_data_points"]}')
            score -= 0.3
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f'Missing required columns: {missing_columns}')
            score -= 0.4
        
        # Check for missing values
        if 'close' in data.columns:
            missing_ratio = data['close'].isna().sum() / len(data)
            if missing_ratio > self.validation_params['max_missing_ratio']:
                issues.append(f'High missing value ratio: {missing_ratio:.3f} > {self.validation_params["max_missing_ratio"]}')
                score -= 0.2
        
        # Check price consistency
        if all(col in data.columns for col in required_columns):
            # Check for non-positive prices
            non_positive = (data[required_columns] <= 0).any().any()
            if non_positive:
                issues.append('Non-positive prices detected')
                score -= 0.3
            
            # Check OHLC consistency
            ohlc_issues = self._check_ohlc_consistency(data)
            if ohlc_issues:
                issues.extend(ohlc_issues)
                score -= 0.2
        
        # Check for extreme price movements
        if 'close' in data.columns:
            price_changes = data['close'].pct_change().abs()
            extreme_moves = (price_changes > self.validation_params['max_price_change']).sum()
            if extreme_moves > 0:
                issues.append(f'Extreme price movements detected: {extreme_moves} moves > {self.validation_params["max_price_change"]*100:.1f}%')
                score -= 0.1
        
        passed = score >= 0.7
        severity = 'error' if not passed else ('warning' if score < 0.9 else 'info')
        
        return ValidationResult(
            passed=passed,
            score=score,
            message=f'Data quality validation: {"PASSED" if passed else "FAILED"}',
            details={'issues': issues, 'data_shape': data.shape},
            severity=severity
        )
    
    def _check_ohlc_consistency(self, data: pd.DataFrame) -> List[str]:
        """Check OHLC consistency."""
        issues = []
        
        # High should be >= max(open, close)
        high_consistent = (data['high'] >= np.maximum(data['open'], data['close'])).all()
        if not high_consistent:
            issues.append('OHLC consistency issue: high < max(open, close)')
        
        # Low should be <= min(open, close)
        low_consistent = (data['low'] <= np.minimum(data['open'], data['close'])).all()
        if not low_consistent:
            issues.append('OHLC consistency issue: low > min(open, close)')
        
        return issues
    
    def _validate_labeling_quality(self, original_data: pd.DataFrame, labeled_data: pd.DataFrame) -> ValidationResult:
        """Validate labeling quality."""
        self.logger.info('🏷️ Validating labeling quality...')
        
        issues = []
        score = 1.0
        
        # Check if labels column exists
        if 'label' not in labeled_data.columns:
            issues.append('Label column not found')
            return ValidationResult(
                passed=False,
                score=0.0,
                message='Labeling quality validation: FAILED - No labels found',
                details={'issues': issues},
                severity='critical'
            )
        
        labels = labeled_data['label']
        
        # Check label distribution
        label_counts = labels.value_counts()
        total_labels = len(labels)
        
        if total_labels == 0:
            issues.append('No labels generated')
            score = 0.0
        else:
            # Check for extreme imbalance
            if len(label_counts) > 1:
                max_count = label_counts.max()
                min_count = label_counts.min()
                imbalance_ratio = max_count / min_count
                
                if imbalance_ratio > self.validation_params['max_imbalance_ratio']:
                    issues.append(f'Extreme label imbalance: ratio {imbalance_ratio:.1f} > {self.validation_params["max_imbalance_ratio"]}')
                    score -= 0.3
            
            # Check minimum label ratio
            for label, count in label_counts.items():
                ratio = count / total_labels
                if ratio < self.validation_params['min_label_ratio']:
                    issues.append(f'Very few samples for label {label}: {ratio:.3f} < {self.validation_params["min_label_ratio"]}')
                    score -= 0.2
        
        # Check for invalid labels
        valid_labels = labels.isin([-1, 0, 1])
        invalid_count = (~valid_labels).sum()
        if invalid_count > 0:
            issues.append(f'Invalid labels detected: {invalid_count} labels not in [-1, 0, 1]')
            score -= 0.4
        
        # Check label consistency with price movements (basic check)
        if 'potential_profit_pct' in labeled_data.columns:
            profits = labeled_data['potential_profit_pct']
            # Check if positive labels have positive profits and vice versa
            positive_labels = labels == 1
            negative_labels = labels == -1
            
            if positive_labels.sum() > 0:
                positive_profits = profits[positive_labels]
                negative_profit_ratio = (positive_profits < 0).sum() / len(positive_profits)
                if negative_profit_ratio > 0.1:  # More than 10% of positive labels have negative profits
                    issues.append(f'Inconsistent positive labels: {negative_profit_ratio:.1%} have negative profits')
                    score -= 0.2
            
            if negative_labels.sum() > 0:
                negative_profits = profits[negative_labels]
                positive_profit_ratio = (negative_profits > 0).sum() / len(negative_profits)
                if positive_profit_ratio > 0.1:  # More than 10% of negative labels have positive profits
                    issues.append(f'Inconsistent negative labels: {positive_profit_ratio:.1%} have positive profits')
                    score -= 0.2
        
        passed = score >= 0.7
        severity = 'error' if not passed else ('warning' if score < 0.9 else 'info')
        
        return ValidationResult(
            passed=passed,
            score=score,
            message=f'Labeling quality validation: {"PASSED" if passed else "FAILED"}',
            details={'issues': issues, 'label_distribution': label_counts.to_dict()},
            severity=severity
        )
    
    def _validate_performance(self, labeled_data: pd.DataFrame) -> ValidationResult:
        """Validate performance metrics."""
        self.logger.info('💰 Validating performance metrics...')
        
        issues = []
        score = 1.0
        
        if 'net_profit_pct' not in labeled_data.columns and 'potential_profit_pct' not in labeled_data.columns:
            issues.append('No profit information available')
            return ValidationResult(
                passed=False,
                score=0.0,
                message='Performance validation: FAILED - No profit data',
                details={'issues': issues},
                severity='critical'
            )
        
        # Get profit data
        profits = labeled_data.get('net_profit_pct', labeled_data.get('potential_profit_pct', pd.Series(dtype=float)))
        labels = labeled_data['label']
        
        if len(profits) == 0:
            issues.append('No profit data available')
            score = 0.0
        else:
            # Calculate performance metrics
            win_rate = (profits > 0).mean()
            avg_profit = profits.mean()
            total_return = profits.sum()
            sharpe_ratio = profits.mean() / profits.std() * np.sqrt(252) if profits.std() > 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = profits.cumsum()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            # Validate win rate
            if win_rate < self.validation_params['min_win_rate']:
                issues.append(f'Low win rate: {win_rate:.3f} < {self.validation_params["min_win_rate"]}')
                score -= 0.3
            
            # Validate Sharpe ratio
            if sharpe_ratio < self.validation_params['min_sharpe_ratio']:
                issues.append(f'Low Sharpe ratio: {sharpe_ratio:.3f} < {self.validation_params["min_sharpe_ratio"]}')
                score -= 0.2
            
            # Validate maximum drawdown
            if max_drawdown > self.validation_params['max_drawdown_threshold']:
                issues.append(f'High maximum drawdown: {max_drawdown:.3f} > {self.validation_params["max_drawdown_threshold"]}')
                score -= 0.2
            
            # Check for consistent losses
            if avg_profit < -0.01:  # Average loss > 1%
                issues.append(f'Consistent losses: average profit {avg_profit:.4f}')
                score -= 0.3
            
            performance_details = {
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(profits)
            }
        
        passed = score >= 0.7
        severity = 'error' if not passed else ('warning' if score < 0.9 else 'info')
        
        return ValidationResult(
            passed=passed,
            score=score,
            message=f'Performance validation: {"PASSED" if passed else "FAILED"}',
            details={'issues': issues, 'performance_metrics': performance_details},
            severity=severity
        )
    
    def _validate_temporal_consistency(self, original_data: pd.DataFrame, labeled_data: pd.DataFrame) -> ValidationResult:
        """Validate temporal consistency (lookahead bias detection)."""
        self.logger.info('⏰ Validating temporal consistency...')
        
        issues = []
        score = 1.0
        
        # Check if indices are properly aligned
        if not original_data.index.equals(labeled_data.index):
            issues.append('Index mismatch between original and labeled data')
            score -= 0.4
        
        # Check for future-looking features (basic check)
        future_features = [col for col in labeled_data.columns 
                          if col.lower().startswith('future_') or col.lower().endswith('_future')]
        if future_features:
            issues.append(f'Potential future-looking features: {future_features}')
            score -= 0.3
        
        # Check temporal ordering
        if isinstance(labeled_data.index, pd.DatetimeIndex):
            if not labeled_data.index.is_monotonic_increasing:
                issues.append('Index is not monotonically increasing')
                score -= 0.2
        
        # Basic lookahead bias check: labels should not be perfectly correlated with future returns
        if len(labeled_data) > 100:
            labels = labeled_data['label'].values
            if 'close' in original_data.columns:
                future_returns = original_data['close'].pct_change().shift(-1).fillna(0).values
                
                # Calculate correlation between labels and future returns
                valid_mask = ~(np.isnan(labels) | np.isnan(future_returns))
                if valid_mask.sum() > 50:
                    correlation = np.corrcoef(labels[valid_mask], future_returns[valid_mask])[0, 1]
                    
                    # High correlation might indicate lookahead bias
                    if abs(correlation) > 0.8:
                        issues.append(f'High correlation with future returns: {correlation:.3f} (possible lookahead bias)')
                        score -= 0.3
        
        passed = score >= 0.7
        severity = 'warning' if not passed else 'info'
        
        return ValidationResult(
            passed=passed,
            score=score,
            message=f'Temporal consistency validation: {"PASSED" if passed else "FAILED"}',
            details={'issues': issues},
            severity=severity
        )
    
    def _validate_statistical_properties(self, labeled_data: pd.DataFrame) -> ValidationResult:
        """Validate statistical properties of the labeling."""
        self.logger.info('📈 Validating statistical properties...')
        
        issues = []
        score = 1.0
        
        if 'net_profit_pct' not in labeled_data.columns and 'potential_profit_pct' not in labeled_data.columns:
            issues.append('No profit information available for statistical validation')
            return ValidationResult(
                passed=False,
                score=0.0,
                message='Statistical validation: FAILED - No profit data',
                details={'issues': issues},
                severity='critical'
            )
        
        profits = labeled_data.get('net_profit_pct', labeled_data.get('potential_profit_pct', pd.Series(dtype=float)))
        
        if len(profits) < 30:  # Need sufficient data for statistical tests
            issues.append(f'Insufficient data for statistical validation: {len(profits)} < 30')
            score -= 0.3
        else:
            # Test for normality of returns
            if len(profits) >= 30:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, p_value = stats.normaltest(profits)
                    
                    if p_value < 0.05:  # Not normally distributed
                        issues.append(f'Returns not normally distributed (p-value: {p_value:.4f})')
                        score -= 0.1
            
            # Check for autocorrelation in returns
            if len(profits) >= 50:
                autocorr = profits.autocorr(lag=1)
                if abs(autocorr) > 0.2:  # High autocorrelation
                    issues.append(f'High autocorrelation in returns: {autocorr:.3f}')
                    score -= 0.1
            
            # Check for volatility clustering (basic check)
            if len(profits) >= 100:
                returns_abs = profits.abs()
                vol_autocorr = returns_abs.autocorr(lag=1)
                if vol_autocorr > 0.3:  # High volatility clustering
                    issues.append(f'Volatility clustering detected: {vol_autocorr:.3f}')
                    score -= 0.1
            
            # Check for extreme outliers
            q99 = profits.quantile(0.99)
            q01 = profits.quantile(0.01)
            extreme_outliers = ((profits > q99) | (profits < q01)).sum()
            if extreme_outliers > len(profits) * 0.05:  # More than 5% extreme outliers
                issues.append(f'High number of extreme outliers: {extreme_outliers} ({extreme_outliers/len(profits)*100:.1f}%)')
                score -= 0.1
        
        passed = score >= 0.7
        severity = 'warning' if not passed else 'info'
        
        return ValidationResult(
            passed=passed,
            score=score,
            message=f'Statistical validation: {"PASSED" if passed else "FAILED"}',
            details={'issues': issues},
            severity=severity
        )
    
    def _generate_validation_report(self, validation_results: Dict[str, ValidationResult]) -> ValidationReport:
        """Generate comprehensive validation report."""
        total_checks = len(validation_results)
        passed_checks = sum(1 for result in validation_results.values() if result.passed)
        failed_checks = total_checks - passed_checks
        warning_checks = sum(1 for result in validation_results.values() if result.severity == 'warning')
        
        # Calculate overall score
        if total_checks > 0:
            overall_score = sum(result.score for result in validation_results.values()) / total_checks
        else:
            overall_score = 0.0
        
        # Identify critical issues
        critical_issues = [
            result.message for result in validation_results.values() 
            if result.severity == 'critical' and not result.passed
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results)
        
        return ValidationReport(
            overall_score=overall_score,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warning_checks=warning_checks,
            critical_issues=critical_issues,
            validation_results=validation_results,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    def _generate_recommendations(self, validation_results: Dict[str, ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for check_name, result in validation_results.items():
            if not result.passed:
                if check_name == 'data_quality':
                    recommendations.append('Improve data quality by fixing missing values and OHLC consistency issues')
                elif check_name == 'labeling_quality':
                    recommendations.append('Adjust labeling parameters to reduce class imbalance and improve label quality')
                elif check_name == 'performance':
                    recommendations.append('Optimize triple barrier parameters to improve win rate and Sharpe ratio')
                elif check_name == 'temporal_consistency':
                    recommendations.append('Review labeling implementation for potential lookahead bias')
                elif check_name == 'statistical_properties':
                    recommendations.append('Consider statistical properties of returns in labeling strategy')
        
        # Add general recommendations
        if not recommendations:
            recommendations.append('All validation checks passed - implementation looks good!')
        
        return recommendations

# Convenience functions
def validate_triple_barrier_implementation(
    data: pd.DataFrame,
    labeled_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None
) -> ValidationReport:
    """Validate a triple barrier labeling implementation.
    
    Args:
        data: Original input data
        labeled_data: Data with triple barrier labels (optional)
        config: Validation configuration
        
    Returns:
        Comprehensive validation report
    """
    validator = TripleBarrierValidator(config)
    return validator.validate_triple_barrier_implementation(data, labeled_data)

def quick_validate_triple_barrier(data: pd.DataFrame, labeled_data: pd.DataFrame) -> bool:
    """Quick validation of triple barrier implementation.
    
    Args:
        data: Original input data
        labeled_data: Data with triple barrier labels
        
    Returns:
        True if validation passes, False otherwise
    """
    report = validate_triple_barrier_implementation(data, labeled_data)
    return report.overall_score >= 0.7 and len(report.critical_issues) == 0

if __name__ == '__main__':
    # Test the validator
    tprint('🧪 Testing Triple Barrier Validator')
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    }, index=dates)
    
    # Create mock labeled data
    labeled_data = data.copy()
    labeled_data['label'] = np.random.choice([-1, 0, 1], 1000, p=[0.3, 0.4, 0.3])
    labeled_data['potential_profit_pct'] = np.random.normal(0.001, 0.005, 1000)
    labeled_data['transaction_cost'] = 0.0008
    
    # Test validation
    tprint('\n🔍 Testing validation...')
    report = validate_triple_barrier_implementation(data, labeled_data)
    
    tprint(f'Validation completed:')
    tprint(f'   → Overall score: {report.overall_score:.3f}')
    tprint(f'   → Passed checks: {report.passed_checks}/{report.total_checks}')
    tprint(f'   → Critical issues: {len(report.critical_issues)}')
    tprint(f'   → Recommendations: {len(report.recommendations)}')
    
    # Test quick validation
    tprint('\n⚡ Testing quick validation...')
    quick_result = quick_validate_triple_barrier(data, labeled_data)
    tprint(f'Quick validation result: {"PASSED" if quick_result else "FAILED"}')
    
    tprint('✅ Triple Barrier Validator test completed successfully!')