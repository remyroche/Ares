"""
Label Quality Assessment and Validation

This module provides comprehensive quality assessment and validation functionality
for triple barrier labels, including statistical analysis, temporal validation,
and cross-validation integration.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Import common utilities
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power,
    validate_finite, validate_positive, validate_range,
    safe_dataframe_operation, validate_dataframe_columns,
    create_summary_statistics
)
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import MathValidation

# Import ML common utilities
from src.utils.ml_common.validation.cv_utils import TemporalCrossValidator, PurgedKFold
from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils

# Setup logging
logger = logging.getLogger(__name__)

class QualityMetric(Enum):
    """Types of quality metrics."""
    LABEL_DISTRIBUTION = "label_distribution"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    PROFIT_CONSISTENCY = "profit_consistency"
    REGIME_BALANCE = "regime_balance"
    CROSS_VALIDATION = "cross_validation"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    OVERALL_QUALITY = "overall_quality"

class QualityLevel(Enum):
    """Quality levels for assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"

@dataclass
class QualityThresholds:
    """Quality thresholds for assessment."""
    # Overall quality thresholds
    excellent_threshold: float = 0.9
    good_threshold: float = 0.8
    fair_threshold: float = 0.7
    poor_threshold: float = 0.6
    
    # Individual metric thresholds
    min_label_balance: float = 0.3  # Minimum ratio for any label class
    max_label_imbalance: float = 0.7  # Maximum ratio for any label class
    min_temporal_consistency: float = 0.6
    min_profit_consistency: float = 0.5
    min_cv_score: float = 0.6
    min_statistical_significance: float = 0.05
    
    # Sample size requirements
    min_samples_per_label: int = 10
    min_total_samples: int = 100
    min_regime_samples: int = 50

@dataclass
class QualityAssessmentResult:
    """Result of quality assessment."""
    overall_quality: float
    quality_level: QualityLevel
    metric_scores: Dict[str, float]
    detailed_metrics: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]
    assessment_timestamp: datetime
    processing_time: float

class LabelQualityAssessment:
    """Comprehensive label quality assessment and validation."""
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """Initialize the quality assessment system.
        
        Args:
            thresholds: Quality thresholds for assessment
        """
        self.thresholds = thresholds or QualityThresholds()
        self.logger = logging.getLogger(f"{__name__}.LabelQualityAssessment")
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("✅ LabelQualityAssessment initialized successfully")

    def _initialize_components(self):
        """Initialize assessment components."""
        try:
            # Initialize common utilities
            self.common_utils = CommonUtilities()
            self.math_validator = MathValidation()
            
            # Initialize evaluation utilities
            self.evaluation_utils = EvaluationUtils()
            
            # Initialize cross-validation components
            self.temporal_cv = TemporalCrossValidator(n_splits=5, purged_pct=0.01)
            self.purged_kfold = PurgedKFold(n_splits=5, purged_pct=0.01)
            
            self.logger.info("✅ Quality assessment components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize quality assessment components: {e}")
            raise

    def assess_quality(
        self,
        labels_df: pd.DataFrame,
        original_data: Optional[pd.DataFrame] = None,
        regime_data: Optional[pd.DataFrame] = None
    ) -> QualityAssessmentResult:
        """Comprehensive quality assessment of labels.
        
        Args:
            labels_df: DataFrame with labels and metadata
            original_data: Original market data (optional)
            regime_data: Regime information (optional)
            
        Returns:
            QualityAssessmentResult with detailed assessment
        """
        start_time = time.time()
        self.logger.info("🔍 Starting comprehensive quality assessment")
        
        warnings_list = []
        errors_list = []
        recommendations = []
        
        try:
            # Validate input
            self._validate_labels_data(labels_df)
            
            # Calculate individual metrics
            metric_scores = {}
            detailed_metrics = {}
            
            # Label distribution analysis
            self.logger.debug("📊 Analyzing label distribution...")
            dist_metrics = self._analyze_label_distribution(labels_df)
            metric_scores[QualityMetric.LABEL_DISTRIBUTION.value] = dist_metrics['score']
            detailed_metrics['label_distribution'] = dist_metrics
            
            # Temporal consistency analysis
            self.logger.debug("⏱️ Analyzing temporal consistency...")
            temp_metrics = self._analyze_temporal_consistency(labels_df)
            metric_scores[QualityMetric.TEMPORAL_CONSISTENCY.value] = temp_metrics['score']
            detailed_metrics['temporal_consistency'] = temp_metrics
            
            # Profit consistency analysis
            self.logger.debug("💰 Analyzing profit consistency...")
            profit_metrics = self._analyze_profit_consistency(labels_df)
            metric_scores[QualityMetric.PROFIT_CONSISTENCY.value] = profit_metrics['score']
            detailed_metrics['profit_consistency'] = profit_metrics
            
            # Regime balance analysis (if applicable)
            if 'regime' in labels_df.columns:
                self.logger.debug("🎯 Analyzing regime balance...")
                regime_metrics = self._analyze_regime_balance(labels_df)
                metric_scores[QualityMetric.REGIME_BALANCE.value] = regime_metrics['score']
                detailed_metrics['regime_balance'] = regime_metrics
            
            # Cross-validation analysis
            self.logger.debug("🔄 Performing cross-validation analysis...")
            cv_metrics = self._analyze_cross_validation(labels_df, original_data)
            metric_scores[QualityMetric.CROSS_VALIDATION.value] = cv_metrics['score']
            detailed_metrics['cross_validation'] = cv_metrics
            
            # Statistical significance analysis
            self.logger.debug("📈 Analyzing statistical significance...")
            stat_metrics = self._analyze_statistical_significance(labels_df)
            metric_scores[QualityMetric.STATISTICAL_SIGNIFICANCE.value] = stat_metrics['score']
            detailed_metrics['statistical_significance'] = stat_metrics
            
            # Calculate overall quality score
            overall_quality = self._calculate_overall_quality(metric_scores)
            quality_level = self._determine_quality_level(overall_quality)
            
            # Generate warnings and recommendations
            warnings_list, recommendations = self._generate_assessment_feedback(
                metric_scores, detailed_metrics, overall_quality
            )
            
            processing_time = time.time() - start_time
            
            result = QualityAssessmentResult(
                overall_quality=overall_quality,
                quality_level=quality_level,
                metric_scores=metric_scores,
                detailed_metrics=detailed_metrics,
                warnings=warnings_list,
                errors=errors_list,
                recommendations=recommendations,
                assessment_timestamp=datetime.now(),
                processing_time=processing_time
            )
            
            self.logger.info(f"✅ Quality assessment completed in {processing_time:.3f}s")
            self.logger.info(f"🎯 Overall quality: {overall_quality:.3f} ({quality_level.value})")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Quality assessment failed after {processing_time:.3f}s: {e}"
            self.logger.error(f"❌ {error_msg}")
            
            errors_list.append(error_msg)
            
            return QualityAssessmentResult(
                overall_quality=0.0,
                quality_level=QualityLevel.FAILED,
                metric_scores={},
                detailed_metrics={},
                warnings=warnings_list,
                errors=errors_list,
                recommendations=[],
                assessment_timestamp=datetime.now(),
                processing_time=processing_time
            )

    def _analyze_label_distribution(self, labels_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze label distribution quality."""
        try:
            label_counts = labels_df['label'].value_counts()
            total_labels = len(labels_df)
            
            # Calculate distribution ratios
            distribution = {
                'positive': safe_divide(label_counts.get(1, 0), total_labels),
                'negative': safe_divide(label_counts.get(-1, 0), total_labels),
                'neutral': safe_divide(label_counts.get(0, 0), total_labels)
            }
            
            # Calculate balance score
            pos_neg_balance = 1.0 - abs(distribution['positive'] - distribution['negative'])
            min_class_ratio = min(distribution.values())
            max_class_ratio = max(distribution.values())
            
            # Check for extreme imbalance
            is_balanced = min_class_ratio >= self.thresholds.min_label_balance
            is_not_extreme = max_class_ratio <= self.thresholds.max_label_imbalance
            
            # Calculate score
            balance_score = pos_neg_balance * 0.5 + (1.0 if is_balanced else 0.0) * 0.3 + (1.0 if is_not_extreme else 0.0) * 0.2
            
            return {
                'score': balance_score,
                'distribution': distribution,
                'label_counts': label_counts.to_dict(),
                'pos_neg_balance': pos_neg_balance,
                'min_class_ratio': min_class_ratio,
                'max_class_ratio': max_class_ratio,
                'is_balanced': is_balanced,
                'is_not_extreme': is_not_extreme
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze label distribution: {e}")
            return {'score': 0.0, 'error': str(e)}

    def _analyze_temporal_consistency(self, labels_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal consistency of labels."""
        try:
            labels = labels_df['label'].values
            
            # Calculate label transitions
            transitions = np.diff(labels)
            
            # Count consistent vs inconsistent transitions
            consistent_transitions = np.sum(transitions == 0)
            total_transitions = len(transitions)
            consistency_ratio = safe_divide(consistent_transitions, total_transitions)
            
            # Calculate transition entropy
            transition_counts = np.bincount(transitions + 2)  # Shift to positive indices
            transition_probs = transition_counts / np.sum(transition_counts)
            entropy = -np.sum(transition_probs * np.log(transition_probs + 1e-10))
            max_entropy = np.log(3)  # 3 possible transitions: -1, 0, 1
            normalized_entropy = safe_divide(entropy, max_entropy)
            
            # Calculate score
            score = consistency_ratio * 0.7 + (1.0 - normalized_entropy) * 0.3
            
            return {
                'score': score,
                'consistency_ratio': consistency_ratio,
                'total_transitions': total_transitions,
                'consistent_transitions': consistent_transitions,
                'entropy': entropy,
                'normalized_entropy': normalized_entropy
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze temporal consistency: {e}")
            return {'score': 0.0, 'error': str(e)}

    def _analyze_profit_consistency(self, labels_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze profit consistency of labels."""
        try:
            profits = labels_df['profit_pct'].values
            
            # Separate positive and negative profits
            positive_profits = profits[profits > 0]
            negative_profits = profits[profits < 0]
            
            if len(positive_profits) == 0 and len(negative_profits) == 0:
                return {'score': 0.0, 'error': 'No profit data available'}
            
            # Calculate consistency metrics
            consistency_scores = []
            
            if len(positive_profits) > 1:
                pos_cv = safe_divide(np.std(positive_profits), np.mean(positive_profits))
                pos_consistency = max(0.0, 1.0 - min(1.0, pos_cv))
                consistency_scores.append(pos_consistency)
            
            if len(negative_profits) > 1:
                neg_cv = safe_divide(np.std(negative_profits), np.mean(negative_profits))
                neg_consistency = max(0.0, 1.0 - min(1.0, neg_cv))
                consistency_scores.append(neg_consistency)
            
            # Overall consistency
            overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
            
            # Profit distribution analysis
            profit_stats = {
                'mean': np.mean(profits),
                'std': np.std(profits),
                'min': np.min(profits),
                'max': np.max(profits),
                'positive_ratio': safe_divide(np.sum(profits > 0), len(profits)),
                'negative_ratio': safe_divide(np.sum(profits < 0), len(profits))
            }
            
            return {
                'score': overall_consistency,
                'consistency_scores': consistency_scores,
                'profit_stats': profit_stats,
                'positive_profits_count': len(positive_profits),
                'negative_profits_count': len(negative_profits)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze profit consistency: {e}")
            return {'score': 0.0, 'error': str(e)}

    def _analyze_regime_balance(self, labels_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime balance in labels."""
        try:
            regime_counts = labels_df['regime'].value_counts()
            total_labels = len(labels_df)
            
            # Calculate regime distribution
            regime_distribution = {
                regime: safe_divide(count, total_labels) 
                for regime, count in regime_counts.items()
            }
            
            # Check if regimes are balanced
            regime_ratios = list(regime_distribution.values())
            min_regime_ratio = min(regime_ratios)
            max_regime_ratio = max(regime_ratios)
            
            # Calculate balance score
            balance_score = min_regime_ratio / max_regime_ratio if max_regime_ratio > 0 else 0.0
            
            # Check minimum samples per regime
            min_samples_per_regime = min(regime_counts.values())
            has_sufficient_samples = min_samples_per_regime >= self.thresholds.min_regime_samples
            
            # Overall score
            score = balance_score * 0.7 + (1.0 if has_sufficient_samples else 0.0) * 0.3
            
            return {
                'score': score,
                'regime_distribution': regime_distribution,
                'regime_counts': regime_counts.to_dict(),
                'balance_score': balance_score,
                'min_samples_per_regime': min_samples_per_regime,
                'has_sufficient_samples': has_sufficient_samples
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze regime balance: {e}")
            return {'score': 0.0, 'error': str(e)}

    def _analyze_cross_validation(self, labels_df: pd.DataFrame, original_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze cross-validation performance."""
        try:
            if original_data is None:
                return {'score': 0.0, 'error': 'No original data provided for cross-validation'}
            
            # Prepare features and labels
            feature_cols = ['open', 'high', 'low', 'close', 'volume'] if 'volume' in original_data.columns else ['open', 'high', 'low', 'close']
            X = original_data[feature_cols].fillna(method='ffill').fillna(0)
            y = labels_df['label']
            
            # Align indices
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            if len(X) < 50:  # Need minimum samples for CV
                return {'score': 0.0, 'error': 'Insufficient data for cross-validation'}
            
            # Perform temporal cross-validation
            try:
                cv_scores = self.temporal_cv.cross_validate(X, y)
                mean_cv_score = np.mean(cv_scores)
                std_cv_score = np.std(cv_scores)
                
                # Calculate score based on CV performance
                score = min(1.0, max(0.0, mean_cv_score))
                
                return {
                    'score': score,
                    'cv_scores': cv_scores,
                    'mean_cv_score': mean_cv_score,
                    'std_cv_score': std_cv_score,
                    'cv_passed': mean_cv_score >= self.thresholds.min_cv_score
                }
                
            except Exception as cv_e:
                self.logger.warning(f"⚠️ Temporal CV failed: {cv_e}")
                
                # Fallback to regular cross-validation
                from sklearn.model_selection import cross_val_score
                from sklearn.ensemble import RandomForestClassifier
                
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                cv_scores = cross_val_score(model, X, y, cv=3)
                mean_cv_score = np.mean(cv_scores)
                
                return {
                    'score': min(1.0, max(0.0, mean_cv_score)),
                    'cv_scores': cv_scores.tolist(),
                    'mean_cv_score': mean_cv_score,
                    'std_cv_score': np.std(cv_scores),
                    'cv_passed': mean_cv_score >= self.thresholds.min_cv_score,
                    'warning': 'Used fallback CV method'
                }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze cross-validation: {e}")
            return {'score': 0.0, 'error': str(e)}

    def _analyze_statistical_significance(self, labels_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze statistical significance of label patterns."""
        try:
            labels = labels_df['label'].values
            profits = labels_df['profit_pct'].values
            
            # Test if labels are significantly different from random
            label_counts = np.bincount(labels + 1)  # Shift to positive indices
            total_labels = len(labels)
            
            # Chi-square test for label distribution
            expected_counts = np.full(3, total_labels / 3)  # Expected uniform distribution
            chi2_stat, p_value = stats.chisquare(label_counts, expected_counts)
            
            # Test if profits are significantly different from zero
            if len(profits) > 1:
                t_stat, t_p_value = stats.ttest_1samp(profits, 0)
            else:
                t_stat, t_p_value = 0, 1
            
            # Calculate significance score
            chi2_significant = p_value < self.thresholds.min_statistical_significance
            t_significant = t_p_value < self.thresholds.min_statistical_significance
            
            significance_score = (1.0 if chi2_significant else 0.0) * 0.5 + (1.0 if t_significant else 0.0) * 0.5
            
            return {
                'score': significance_score,
                'chi2_stat': chi2_stat,
                'chi2_p_value': p_value,
                't_stat': t_stat,
                't_p_value': t_p_value,
                'chi2_significant': chi2_significant,
                't_significant': t_significant
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze statistical significance: {e}")
            return {'score': 0.0, 'error': str(e)}

    def _calculate_overall_quality(self, metric_scores: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        try:
            # Weighted average of all metrics
            weights = {
                QualityMetric.LABEL_DISTRIBUTION.value: 0.25,
                QualityMetric.TEMPORAL_CONSISTENCY.value: 0.20,
                QualityMetric.PROFIT_CONSISTENCY.value: 0.20,
                QualityMetric.REGIME_BALANCE.value: 0.15,
                QualityMetric.CROSS_VALIDATION.value: 0.15,
                QualityMetric.STATISTICAL_SIGNIFICANCE.value: 0.05
            }
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for metric, score in metric_scores.items():
                weight = weights.get(metric, 0.0)
                weighted_sum += score * weight
                total_weight += weight
            
            overall_quality = safe_divide(weighted_sum, total_weight)
            return max(0.0, min(1.0, overall_quality))
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate overall quality: {e}")
            return 0.0

    def _determine_quality_level(self, overall_quality: float) -> QualityLevel:
        """Determine quality level based on overall score."""
        if overall_quality >= self.thresholds.excellent_threshold:
            return QualityLevel.EXCELLENT
        elif overall_quality >= self.thresholds.good_threshold:
            return QualityLevel.GOOD
        elif overall_quality >= self.thresholds.fair_threshold:
            return QualityLevel.FAIR
        elif overall_quality >= self.thresholds.poor_threshold:
            return QualityLevel.POOR
        else:
            return QualityLevel.FAILED

    def _generate_assessment_feedback(
        self, 
        metric_scores: Dict[str, float], 
        detailed_metrics: Dict[str, Any], 
        overall_quality: float
    ) -> Tuple[List[str], List[str]]:
        """Generate warnings and recommendations based on assessment."""
        warnings = []
        recommendations = []
        
        # Check individual metrics
        for metric, score in metric_scores.items():
            if score < 0.5:
                warnings.append(f"Low {metric} score: {score:.3f}")
                
                if metric == QualityMetric.LABEL_DISTRIBUTION.value:
                    recommendations.append("Consider adjusting barrier parameters to improve label balance")
                elif metric == QualityMetric.TEMPORAL_CONSISTENCY.value:
                    recommendations.append("Labels may be too noisy - consider smoothing or different parameters")
                elif metric == QualityMetric.PROFIT_CONSISTENCY.value:
                    recommendations.append("Profit patterns are inconsistent - review barrier calculations")
                elif metric == QualityMetric.CROSS_VALIDATION.value:
                    recommendations.append("Labels may not be predictive - consider feature engineering")
        
        # Overall quality feedback
        if overall_quality < self.thresholds.poor_threshold:
            warnings.append(f"Overall quality is very low: {overall_quality:.3f}")
            recommendations.append("Consider completely re-evaluating labeling strategy")
        elif overall_quality < self.thresholds.fair_threshold:
            warnings.append(f"Overall quality is below fair threshold: {overall_quality:.3f}")
            recommendations.append("Consider adjusting labeling parameters or data preprocessing")
        
        return warnings, recommendations

    def _validate_labels_data(self, labels_df: pd.DataFrame):
        """Validate labels data for assessment."""
        required_columns = ['label', 'profit_pct']
        missing_columns = [col for col in required_columns if col not in labels_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns for quality assessment: {missing_columns}")
        
        if len(labels_df) < self.thresholds.min_total_samples:
            raise ValueError(f"Insufficient samples for quality assessment: {len(labels_df)} < {self.thresholds.min_total_samples}")

    def generate_quality_report(self, result: QualityAssessmentResult) -> str:
        """Generate a detailed quality report."""
        report = []
        report.append("=" * 60)
        report.append("TRIPLE BARRIER LABEL QUALITY ASSESSMENT REPORT")
        report.append("=" * 60)
        report.append(f"Assessment Time: {result.assessment_timestamp}")
        report.append(f"Processing Time: {result.processing_time:.3f} seconds")
        report.append("")
        
        # Overall quality
        report.append(f"OVERALL QUALITY: {result.overall_quality:.3f} ({result.quality_level.value.upper()})")
        report.append("")
        
        # Individual metrics
        report.append("INDIVIDUAL METRICS:")
        report.append("-" * 30)
        for metric, score in result.metric_scores.items():
            report.append(f"{metric.replace('_', ' ').title()}: {score:.3f}")
        report.append("")
        
        # Warnings
        if result.warnings:
            report.append("WARNINGS:")
            report.append("-" * 30)
            for warning in result.warnings:
                report.append(f"⚠️ {warning}")
            report.append("")
        
        # Recommendations
        if result.recommendations:
            report.append("RECOMMENDATIONS:")
            report.append("-" * 30)
            for i, rec in enumerate(result.recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # Errors
        if result.errors:
            report.append("ERRORS:")
            report.append("-" * 30)
            for error in result.errors:
                report.append(f"❌ {error}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)