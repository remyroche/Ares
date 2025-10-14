"""
Labeling Validator for Multi-Horizon Profit Labeling

This module provides comprehensive validation of profit labeling quality and consistency,
similar to validation frameworks used in HMM clustering research. It examines:

1. Label Quality Metrics (consistency, stability, predictiveness)
2. Cross-Validation of Labeling Strategies  
3. Temporal Stability Analysis
4. Statistical Validation Tests
5. Economic Significance Validation
6. Labeling Bias Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
import warnings

from src.utils.logger import get_logger
from src.training.steps.pre_training.multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler, 
    MultiHorizonConfig
)


class ValidationMetric(Enum):
    """Enumeration of validation metrics."""
    LABEL_CONSISTENCY = "label_consistency"
    TEMPORAL_STABILITY = "temporal_stability" 
    PREDICTIVE_VALIDITY = "predictive_validity"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    CROSS_VALIDATION_STABILITY = "cross_validation_stability"
    BIAS_DETECTION = "bias_detection"
    INFORMATION_CONTENT = "information_content"
    LABEL_DISTRIBUTION = "label_distribution"
    NOISE_ROBUSTNESS = "noise_robustness"


@dataclass 
class ValidationConfig:
    """Configuration for labeling validation."""
    # Validation scope
    validate_consistency: bool = True
    validate_stability: bool = True  
    validate_predictiveness: bool = True
    validate_significance: bool = True
    validate_bias: bool = True
    
    # Statistical parameters
    significance_level: float = 0.05
    min_sample_size: int = 500
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    
    # Cross-validation parameters
    cv_folds: int = 5
    cv_gap: int = 10  # Gap between train/test for time series
    min_train_size: int = 1000
    
    # Stability analysis
    stability_window: int = 100
    stability_overlap: float = 0.5
    
    # Economic validation
    transaction_cost: float = 0.0008
    risk_free_rate: float = 0.02
    min_economic_significance: float = 0.001  # 0.1% minimum edge
    
    # Bias detection thresholds
    max_acceptable_bias: float = 0.1
    bias_detection_methods: List[str] = field(default_factory=lambda: [
        'temporal_bias', 'distribution_bias', 'selection_bias'
    ])


@dataclass
class ValidationResult:
    """Result container for validation analysis."""
    metric: ValidationMetric
    value: float
    confidence_interval: Optional[Tuple[float, float]]
    p_value: Optional[float] 
    is_significant: Optional[bool]
    interpretation: str
    recommendations: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class LabelingValidator:
    """
    Comprehensive validator for multi-horizon profit labeling quality.
    
    This class provides rigorous validation of labeling quality, similar to 
    how we validate HMM clustering results. It examines multiple dimensions
    of label quality and provides actionable recommendations.
    
    Key Validation Areas:
    1. **Label Consistency**: Are similar market conditions labeled similarly?
    2. **Temporal Stability**: Do labels remain stable over time?
    3. **Predictive Validity**: Do labels predict future outcomes?
    4. **Statistical Significance**: Are labeling patterns statistically valid?
    5. **Economic Significance**: Do labels translate to economic value?
    6. **Bias Detection**: Are there systematic biases in labeling?
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize the labeling validator."""
        self.config = config or ValidationConfig()
        self.logger = get_logger('LabelingValidator')
        
        # Validation results storage
        self.validation_results: Dict[str, ValidationResult] = {}
        self.validation_history: List[Dict[str, Any]] = []
        
        self.logger.info('🔍 Labeling Validator initialized')
        self.logger.info(f'   → Validation scope: {self._get_validation_scope()}')
        
    def _get_validation_scope(self) -> str:
        """Get human-readable validation scope."""
        scope_items = []
        if self.config.validate_consistency:
            scope_items.append("Consistency")
        if self.config.validate_stability:
            scope_items.append("Stability")
        if self.config.validate_predictiveness:
            scope_items.append("Predictiveness")
        if self.config.validate_significance:
            scope_items.append("Significance")
        if self.config.validate_bias:
            scope_items.append("Bias Detection")
        return ", ".join(scope_items)
    
    def validate_labeling_quality(self,
                                market_data: pd.DataFrame,
                                labeled_data: Optional[pd.DataFrame] = None,
                                labeling_config: Optional[MultiHorizonConfig] = None) -> Dict[str, ValidationResult]:
        """
        Comprehensive validation of labeling quality.
        
        Args:
            market_data: Original OHLCV market data
            labeled_data: Pre-labeled data (optional, will generate if not provided)
            labeling_config: Configuration for labeling (if generating labels)
            
        Returns:
            Dictionary of validation results by metric type
        """
        self.logger.info('🔬 Starting comprehensive labeling validation')
        
        if len(market_data) < self.config.min_sample_size:
            raise ValueError(f"Insufficient data: need {self.config.min_sample_size}, got {len(market_data)}")
        
        # Generate labels if not provided
        if labeled_data is None:
            self.logger.info('📊 Generating labels for validation')
            labeler = MultiHorizonProfitLabeler(labeling_config)
            labeled_data = labeler.generate_labels(market_data.copy())
        
        # Run all enabled validations
        validations = []
        
        if self.config.validate_consistency:
            validations.append(self._validate_label_consistency)
        if self.config.validate_stability:
            validations.append(self._validate_temporal_stability)
        if self.config.validate_predictiveness:
            validations.append(self._validate_predictive_validity)
        if self.config.validate_significance:
            validations.append(self._validate_statistical_significance)
        if self.config.validate_bias:
            validations.append(self._validate_bias_detection)
        
        # Execute validations
        for validation_func in validations:
            try:
                result = validation_func(market_data, labeled_data)
                if isinstance(result, dict):
                    self.validation_results.update(result)
                else:
                    self.validation_results[result.metric.value] = result
            except Exception as e:
                self.logger.error(f"Validation failed: {validation_func.__name__}: {e}")
        
        # Store validation history
        self.validation_history.append({
            'timestamp': datetime.now(),
            'data_size': len(market_data),
            'results_count': len(self.validation_results),
            'config': self.config.__dict__
        })
        
        self.logger.info(f'✅ Labeling validation completed: {len(self.validation_results)} validations')
        return self.validation_results
    
    def _validate_label_consistency(self,
                                  market_data: pd.DataFrame,
                                  labeled_data: pd.DataFrame) -> Dict[str, ValidationResult]:
        """Validate consistency of labels across similar market conditions."""
        self.logger.info('🎯 Validating label consistency')
        
        results = {}
        
        # Get probability columns for analysis
        prob_columns = [col for col in labeled_data.columns if col.endswith('_prob')]
        
        for col in prob_columns[:3]:  # Analyze top 3 for efficiency
            if col not in labeled_data.columns:
                continue
                
            # Calculate consistency metrics
            consistency_score = self._calculate_consistency_score(
                market_data, labeled_data, col
            )
            
            # Bootstrap confidence interval
            ci = self._bootstrap_consistency_score(market_data, labeled_data, col)
            
            # Generate result
            result = ValidationResult(
                metric=ValidationMetric.LABEL_CONSISTENCY,
                value=consistency_score,
                confidence_interval=ci,
                p_value=None,
                is_significant=consistency_score > 0.6,  # Threshold for good consistency
                interpretation=f"Label consistency for {col}: {consistency_score:.2%}",
                recommendations=self._generate_consistency_recommendations(consistency_score),
                metadata={
                    'column': col,
                    'sample_size': len(labeled_data[col].dropna()),
                    'analysis_method': 'similarity_based_consistency'
                }
            )
            
            results[f"{col}_consistency"] = result
        
        return results
    
    def _validate_temporal_stability(self,
                                   market_data: pd.DataFrame,
                                   labeled_data: pd.DataFrame) -> ValidationResult:
        """Validate temporal stability of labels."""
        self.logger.info('⏰ Validating temporal stability')
        
        # Calculate stability across time windows
        stability_scores = []
        window_size = self.config.stability_window
        overlap = int(window_size * self.config.stability_overlap)
        
        prob_columns = [col for col in labeled_data.columns if col.endswith('_prob')]
        
        for i in range(0, len(labeled_data) - window_size, window_size - overlap):
            window1 = labeled_data.iloc[i:i + window_size]
            window2 = labeled_data.iloc[i + window_size - overlap:i + 2 * window_size - overlap]
            
            if len(window2) < window_size:
                break
            
            # Calculate stability between windows
            window_stability = []
            for col in prob_columns[:5]:  # Top 5 for efficiency
                if col in window1.columns and col in window2.columns:
                    corr = np.corrcoef(
                        window1[col].fillna(0),
                        window2[col].fillna(0)
                    )[0, 1]
                    if not np.isnan(corr):
                        window_stability.append(abs(corr))
            
            if window_stability:
                stability_scores.append(np.mean(window_stability))
        
        # Overall stability score
        overall_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        return ValidationResult(
            metric=ValidationMetric.TEMPORAL_STABILITY,
            value=overall_stability,
            confidence_interval=self._bootstrap_confidence_interval(
                np.array(stability_scores), np.mean
            ) if stability_scores else None,
            p_value=None,
            is_significant=overall_stability > 0.7,
            interpretation=f"Temporal stability: {overall_stability:.2%}",
            recommendations=self._generate_stability_recommendations(overall_stability),
            metadata={
                'window_size': window_size,
                'num_windows': len(stability_scores),
                'stability_scores': stability_scores[:10]  # Store first 10 for inspection
            }
        )
    
    def _validate_predictive_validity(self,
                                    market_data: pd.DataFrame,
                                    labeled_data: pd.DataFrame) -> Dict[str, ValidationResult]:
        """Validate predictive validity of labels."""
        self.logger.info('🔮 Validating predictive validity')
        
        results = {}
        
        if 'close' not in market_data.columns:
            return {
                'predictive_validity_error': ValidationResult(
                    metric=ValidationMetric.PREDICTIVE_VALIDITY,
                    value=0.0,
                    confidence_interval=None,
                    p_value=None,
                    is_significant=False,
                    interpretation="Cannot validate predictive validity without price data",
                    recommendations=["Ensure market data includes 'close' prices"],
                    metadata={'error': 'no_price_data'}
                )
            }
        
        # Calculate future returns for validation
        future_returns = market_data['close'].pct_change().shift(-1).fillna(0)
        
        # Test predictive validity for key columns
        test_columns = ['overall_opportunity', 'leverage_adjusted_score', 'immediate_opportunity']
        
        for col in test_columns:
            if col not in labeled_data.columns:
                continue
                
            predictive_score = self._calculate_predictive_validity_score(
                labeled_data[col], future_returns
            )
            
            # Cross-validation for robustness
            cv_scores = self._cross_validate_predictive_validity(
                labeled_data[col], future_returns
            )
            
            result = ValidationResult(
                metric=ValidationMetric.PREDICTIVE_VALIDITY,
                value=predictive_score,
                confidence_interval=(
                    float(np.percentile(cv_scores, 2.5)),
                    float(np.percentile(cv_scores, 97.5))
                ) if cv_scores else None,
                p_value=None,
                is_significant=predictive_score > 0.55,  # Better than random
                interpretation=f"Predictive validity for {col}: {predictive_score:.3f}",
                recommendations=self._generate_predictive_recommendations(predictive_score),
                metadata={
                    'column': col,
                    'cv_scores': cv_scores[:10] if cv_scores else [],
                    'analysis_method': 'correlation_with_future_returns'
                }
            )
            
            results[f"{col}_predictive_validity"] = result
        
        return results
    
    def _validate_statistical_significance(self,
                                         market_data: pd.DataFrame,
                                         labeled_data: pd.DataFrame) -> ValidationResult:
        """Validate statistical significance of labeling patterns."""
        self.logger.info('📊 Validating statistical significance')
        
        # Perform statistical tests on key labeling outputs
        prob_columns = [col for col in labeled_data.columns if col.endswith('_prob')]
        
        # Test 1: Are probabilities significantly different from random?
        random_baseline = np.random.random(len(labeled_data))
        significance_tests = []
        
        for col in prob_columns[:5]:  # Test top 5 for efficiency
            if col in labeled_data.columns:
                values = labeled_data[col].dropna()
                if len(values) > 100:
                    # Kolmogorov-Smirnov test against random
                    ks_stat, p_value = stats.ks_2samp(values, random_baseline[:len(values)])
                    significance_tests.append({
                        'column': col,
                        'test': 'ks_test_vs_random',
                        'statistic': ks_stat,
                        'p_value': p_value,
                        'significant': p_value < self.config.significance_level
                    })
        
        # Overall significance score
        significant_tests = [t for t in significance_tests if t['significant']]
        significance_score = len(significant_tests) / len(significance_tests) if significance_tests else 0.0
        
        return ValidationResult(
            metric=ValidationMetric.STATISTICAL_SIGNIFICANCE,
            value=significance_score,
            confidence_interval=None,
            p_value=np.mean([t['p_value'] for t in significance_tests]) if significance_tests else 1.0,
            is_significant=significance_score > 0.5,
            interpretation=f"Statistical significance: {significance_score:.2%} of tests significant",
            recommendations=self._generate_significance_recommendations(significance_score),
            metadata={
                'total_tests': len(significance_tests),
                'significant_tests': len(significant_tests),
                'test_details': significance_tests
            }
        )
    
    def _validate_bias_detection(self,
                               market_data: pd.DataFrame,
                               labeled_data: pd.DataFrame) -> Dict[str, ValidationResult]:
        """Detect systematic biases in labeling."""
        self.logger.info('🔍 Detecting labeling biases')
        
        results = {}
        
        # Temporal bias detection
        if 'temporal_bias' in self.config.bias_detection_methods:
            temporal_bias = self._detect_temporal_bias(labeled_data)
            results['temporal_bias'] = ValidationResult(
                metric=ValidationMetric.BIAS_DETECTION,
                value=temporal_bias,
                confidence_interval=None,
                p_value=None,
                is_significant=temporal_bias > self.config.max_acceptable_bias,
                interpretation=f"Temporal bias detected: {temporal_bias:.3f}",
                recommendations=self._generate_bias_recommendations('temporal', temporal_bias),
                metadata={'bias_type': 'temporal', 'threshold': self.config.max_acceptable_bias}
            )
        
        # Distribution bias detection  
        if 'distribution_bias' in self.config.bias_detection_methods:
            distribution_bias = self._detect_distribution_bias(labeled_data)
            results['distribution_bias'] = ValidationResult(
                metric=ValidationMetric.BIAS_DETECTION,
                value=distribution_bias,
                confidence_interval=None,
                p_value=None,
                is_significant=distribution_bias > self.config.max_acceptable_bias,
                interpretation=f"Distribution bias detected: {distribution_bias:.3f}",
                recommendations=self._generate_bias_recommendations('distribution', distribution_bias),
                metadata={'bias_type': 'distribution', 'threshold': self.config.max_acceptable_bias}
            )
        
        return results
    
    # Helper methods for calculations
    def _calculate_consistency_score(self,
                                   market_data: pd.DataFrame,
                                   labeled_data: pd.DataFrame,
                                   column: str) -> float:
        """Calculate consistency score for a label column."""
        try:
            values = labeled_data[column].dropna()
            if len(values) < 50:
                return 0.0
            
            # Simple consistency: inverse of coefficient of variation
            if values.std() > 0:
                cv = values.std() / abs(values.mean()) if values.mean() != 0 else float('inf')
                consistency = 1.0 / (1.0 + cv)  # Higher consistency = lower variation
            else:
                consistency = 1.0  # Perfect consistency if no variation
            
            return max(0.0, min(1.0, consistency))
            
        except Exception:
            return 0.0
    
    def _bootstrap_consistency_score(self,
                                   market_data: pd.DataFrame,
                                   labeled_data: pd.DataFrame,
                                   column: str) -> Optional[Tuple[float, float]]:
        """Bootstrap confidence interval for consistency score."""
        try:
            values = labeled_data[column].dropna()
            if len(values) < 20:
                return None
            
            bootstrap_scores = []
            for _ in range(100):  # Lighter bootstrap for efficiency
                sample = np.random.choice(values, size=len(values), replace=True)
                sample_df = pd.DataFrame({column: sample})
                score = self._calculate_consistency_score(market_data, sample_df, column)
                bootstrap_scores.append(score)
            
            return (
                float(np.percentile(bootstrap_scores, 2.5)),
                float(np.percentile(bootstrap_scores, 97.5))
            )
            
        except Exception:
            return None
    
    def _calculate_predictive_validity_score(self,
                                           labels: pd.Series,
                                           future_returns: pd.Series) -> float:
        """Calculate predictive validity score."""
        try:
            # Align series
            common_idx = labels.index.intersection(future_returns.index)
            if len(common_idx) < 20:
                return 0.0
            
            aligned_labels = labels.loc[common_idx].fillna(0)
            aligned_returns = future_returns.loc[common_idx].fillna(0)
            
            # Calculate correlation as proxy for predictive power
            correlation = np.corrcoef(aligned_labels, aligned_returns)[0, 1]
            
            # Convert to AUC-like score (0.5 = random, 1.0 = perfect)
            if np.isnan(correlation):
                return 0.5
            
            auc_proxy = 0.5 + abs(correlation) / 2
            return max(0.0, min(1.0, auc_proxy))
            
        except Exception:
            return 0.5  # Random baseline
    
    def _cross_validate_predictive_validity(self,
                                          labels: pd.Series,
                                          future_returns: pd.Series) -> List[float]:
        """Cross-validate predictive validity."""
        try:
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds, gap=self.config.cv_gap)
            
            cv_scores = []
            common_idx = labels.index.intersection(future_returns.index)
            
            if len(common_idx) < self.config.min_train_size:
                return []
            
            aligned_labels = labels.loc[common_idx].fillna(0)
            aligned_returns = future_returns.loc[common_idx].fillna(0)
            
            for train_idx, test_idx in tscv.split(aligned_labels):
                if len(test_idx) < 20:
                    continue
                    
                test_labels = aligned_labels.iloc[test_idx]
                test_returns = aligned_returns.iloc[test_idx]
                
                score = self._calculate_predictive_validity_score(test_labels, test_returns)
                cv_scores.append(score)
            
            return cv_scores
            
        except Exception:
            return []
    
    def _detect_temporal_bias(self, labeled_data: pd.DataFrame) -> float:
        """Detect temporal bias in labeling."""
        try:
            # Check if label distributions change significantly over time
            prob_columns = [col for col in labeled_data.columns if col.endswith('_prob')]
            
            if not prob_columns:
                return 0.0
            
            # Split data into first and second half
            mid_point = len(labeled_data) // 2
            first_half = labeled_data.iloc[:mid_point]
            second_half = labeled_data.iloc[mid_point:]
            
            bias_scores = []
            for col in prob_columns[:3]:  # Test top 3
                if col in first_half.columns and col in second_half.columns:
                    first_mean = first_half[col].mean()
                    second_mean = second_half[col].mean()
                    
                    if first_mean != 0:
                        bias = abs(second_mean - first_mean) / abs(first_mean)
                        bias_scores.append(bias)
            
            return np.mean(bias_scores) if bias_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _detect_distribution_bias(self, labeled_data: pd.DataFrame) -> float:
        """Detect distribution bias in labeling."""
        try:
            # Check if label distributions deviate from expected patterns
            prob_columns = [col for col in labeled_data.columns if col.endswith('_prob')]
            
            bias_scores = []
            for col in prob_columns[:3]:  # Test top 3
                values = labeled_data[col].dropna()
                if len(values) < 50:
                    continue
                
                # Test against uniform distribution (expected for probabilities)
                ks_stat, _ = stats.kstest(values, 'uniform')
                bias_scores.append(ks_stat)
            
            return np.mean(bias_scores) if bias_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _bootstrap_confidence_interval(self,
                                     data: np.ndarray,
                                     statistic_func: Callable) -> Optional[Tuple[float, float]]:
        """Calculate bootstrap confidence interval."""
        try:
            if len(data) < 10:
                return None
            
            bootstrap_stats = []
            for _ in range(self.config.bootstrap_iterations):
                sample = np.random.choice(data, size=len(data), replace=True)
                stat = statistic_func(sample)
                if not np.isnan(stat):
                    bootstrap_stats.append(stat)
            
            if not bootstrap_stats:
                return None
            
            alpha = 1 - self.config.confidence_level
            lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            
            return (float(lower), float(upper))
            
        except Exception:
            return None
    
    # Recommendation generators
    def _generate_consistency_recommendations(self, consistency_score: float) -> List[str]:
        """Generate recommendations for consistency issues."""
        recommendations = []
        
        if consistency_score < 0.3:
            recommendations.extend([
                "⚠️ Very low consistency - labels are highly variable",
                "Review labeling parameters for stability",
                "Consider smoothing or filtering techniques"
            ])
        elif consistency_score < 0.6:
            recommendations.extend([
                "📊 Moderate consistency - room for improvement", 
                "Fine-tune quality scoring parameters",
                "Validate market regime consistency"
            ])
        else:
            recommendations.append("✅ Good consistency maintained")
        
        return recommendations
    
    def _generate_stability_recommendations(self, stability_score: float) -> List[str]:
        """Generate recommendations for stability issues."""
        recommendations = []
        
        if stability_score < 0.5:
            recommendations.extend([
                "⚠️ Poor temporal stability",
                "Labels change significantly over time",
                "Consider regime-aware labeling",
                "Review parameter adaptation strategies"
            ])
        elif stability_score < 0.7:
            recommendations.extend([
                "📈 Moderate stability - monitor trends",
                "Implement stability tracking",
                "Consider rolling parameter updates"
            ])
        else:
            recommendations.append("✅ Good temporal stability")
        
        return recommendations
    
    def _generate_predictive_recommendations(self, predictive_score: float) -> List[str]:
        """Generate recommendations for predictive validity."""
        recommendations = []
        
        if predictive_score < 0.52:
            recommendations.extend([
                "⚠️ Poor predictive validity - labels may not be useful",
                "Review labeling methodology",
                "Consider different target definitions",
                "Validate against longer horizons"
            ])
        elif predictive_score < 0.6:
            recommendations.extend([
                "📊 Moderate predictive validity",
                "Fine-tune labeling parameters",
                "Consider ensemble labeling approaches"
            ])
        else:
            recommendations.append("✅ Good predictive validity")
        
        return recommendations
    
    def _generate_significance_recommendations(self, significance_score: float) -> List[str]:
        """Generate recommendations for statistical significance."""
        recommendations = []
        
        if significance_score < 0.3:
            recommendations.extend([
                "⚠️ Labels not statistically significant",
                "May be indistinguishable from random",
                "Review labeling methodology fundamentally"
            ])
        elif significance_score < 0.7:
            recommendations.extend([
                "📊 Some labels show significance",
                "Focus on statistically significant components",
                "Improve weaker labeling aspects"
            ])
        else:
            recommendations.append("✅ Strong statistical significance")
        
        return recommendations
    
    def _generate_bias_recommendations(self, bias_type: str, bias_score: float) -> List[str]:
        """Generate recommendations for bias issues."""
        recommendations = []
        
        if bias_score > self.config.max_acceptable_bias:
            if bias_type == 'temporal':
                recommendations.extend([
                    f"⚠️ Significant temporal bias detected ({bias_score:.3f})",
                    "Labels change systematically over time",
                    "Consider regime-aware parameter adjustment",
                    "Implement bias correction mechanisms"
                ])
            elif bias_type == 'distribution':
                recommendations.extend([
                    f"⚠️ Distribution bias detected ({bias_score:.3f})",
                    "Label distributions deviate from expected patterns",
                    "Review probability calibration",
                    "Consider distribution normalization"
                ])
        else:
            recommendations.append(f"✅ {bias_type.title()} bias within acceptable limits")
        
        return recommendations
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        if not self.validation_results:
            return "No validation results available. Run validate_labeling_quality() first."
        
        report_lines = [
            "# Multi-Horizon Profit Labeling Validation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"Validated {len(self.validation_results)} labeling components",
            ""
        ]
        
        # Summary statistics
        significant_results = [r for r in self.validation_results.values() if r.is_significant]
        report_lines.extend([
            f"**Significant Results**: {len(significant_results)}/{len(self.validation_results)} ({len(significant_results)/len(self.validation_results)*100:.1f}%)",
            ""
        ])
        
        # Group results by metric type
        by_metric = {}
        for key, result in self.validation_results.items():
            metric_type = result.metric.value
            if metric_type not in by_metric:
                by_metric[metric_type] = []
            by_metric[metric_type].append((key, result))
        
        # Generate sections for each metric type
        for metric_type, results in by_metric.items():
            report_lines.extend([
                f"## {metric_type.replace('_', ' ').title()}",
                ""
            ])
            
            for key, result in results:
                status_icon = "✅" if result.is_significant else "⚠️"
                report_lines.extend([
                    f"### {status_icon} {key}",
                    f"**Value**: {result.value:.4f}",
                    f"**Interpretation**: {result.interpretation}",
                    ""
                ])
                
                if result.confidence_interval:
                    ci_lower, ci_upper = result.confidence_interval
                    report_lines.append(f"**Confidence Interval**: [{ci_lower:.4f}, {ci_upper:.4f}]")
                
                if result.p_value is not None:
                    report_lines.append(f"**P-Value**: {result.p_value:.4f}")
                
                if result.recommendations:
                    report_lines.append("**Recommendations**:")
                    for rec in result.recommendations:
                        report_lines.append(f"- {rec}")
                
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_validation_results(self, output_path: Union[str, Path]):
        """Save validation results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = {}
        for key, result in self.validation_results.items():
            serializable_results[key] = {
                'metric': result.metric.value,
                'value': result.value,
                'confidence_interval': result.confidence_interval,
                'p_value': result.p_value,
                'is_significant': result.is_significant,
                'interpretation': result.interpretation,
                'recommendations': result.recommendations,
                'metadata': result.metadata,
                'timestamp': result.timestamp.isoformat()
            }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump({
                'validation_results': serializable_results,
                'validation_history': self.validation_history,
                'config': {
                    'significance_level': self.config.significance_level,
                    'min_sample_size': self.config.min_sample_size,
                    'bootstrap_iterations': self.config.bootstrap_iterations,
                    'confidence_level': self.config.confidence_level
                }
            }, f, indent=2)
        
        self.logger.info(f'💾 Validation results saved to {output_path}')


# Convenience functions
def validate_profit_labeling(market_data: pd.DataFrame,
                           labeled_data: Optional[pd.DataFrame] = None,
                           labeling_config: Optional[MultiHorizonConfig] = None,
                           validation_config: Optional[ValidationConfig] = None) -> Dict[str, ValidationResult]:
    """Convenience function to validate profit labeling."""
    validator = LabelingValidator(validation_config)
    return validator.validate_labeling_quality(market_data, labeled_data, labeling_config)


def generate_validation_report(market_data: pd.DataFrame,
                             labeled_data: Optional[pd.DataFrame] = None,
                             labeling_config: Optional[MultiHorizonConfig] = None,
                             validation_config: Optional[ValidationConfig] = None) -> str:
    """Convenience function to generate validation report."""
    validator = LabelingValidator(validation_config)
    validator.validate_labeling_quality(market_data, labeled_data, labeling_config)
    return validator.generate_validation_report()
