"""
Pattern Validation Module

This module provides comprehensive validation for discovered price patterns,
including statistical significance testing, economic relevance validation,
frequency analysis, and predictability scoring.

Consolidated from multiple validation approaches across the research framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.utils.logger import system_logger


class ValidationMetric(Enum):
    """Types of pattern validation metrics."""
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    FREQUENCY_ANALYSIS = "frequency_analysis"
    PREDICTABILITY_SCORE = "predictability_score"
    TEMPORAL_STABILITY = "temporal_stability"
    MAGNITUDE_CONSISTENCY = "magnitude_consistency"
    DURATION_DISTRIBUTION = "duration_distribution"


@dataclass
class PatternValidationResult:
    """Result of pattern validation analysis."""
    pattern_name: str
    validation_metrics: Dict[ValidationMetric, float]
    statistical_tests: Dict[str, float]
    economic_tests: Dict[str, float]
    temporal_analysis: Dict[str, float]
    is_valid_pattern: bool
    validation_summary: str
    recommendations: List[str]


class PatternValidator:
    """
    Comprehensive pattern validation system.
    
    Validates discovered patterns for statistical significance, economic relevance,
    temporal stability, and trading utility.
    """
    
    def __init__(self):
        self.logger = system_logger.getChild('PatternValidator')
    
    def validate_pattern(self, 
                        pattern_labels: pd.Series,
                        pattern_intensity: pd.Series,
                        price_data: pd.Series,
                        pattern_name: str) -> PatternValidationResult:
        """
        Comprehensive validation of a single pattern.
        
        Args:
            pattern_labels: Binary pattern labels
            pattern_intensity: Pattern intensity values
            price_data: Price series for validation
            pattern_name: Name of the pattern
            
        Returns:
            Complete validation results
        """
        
        self.logger.info(f"🔍 Validating pattern: {pattern_name}")
        
        # Calculate validation metrics
        validation_metrics = {}
        
        # 1. Statistical Significance
        statistical_tests = self._test_statistical_significance(pattern_labels, price_data)
        validation_metrics[ValidationMetric.STATISTICAL_SIGNIFICANCE] = 1.0 - statistical_tests['p_value']
        
        # 2. Economic Significance
        economic_tests = self._test_economic_significance(pattern_labels, price_data)
        validation_metrics[ValidationMetric.ECONOMIC_SIGNIFICANCE] = economic_tests['economic_score']
        
        # 3. Frequency Analysis
        frequency_analysis = self._analyze_pattern_frequency(pattern_labels)
        validation_metrics[ValidationMetric.FREQUENCY_ANALYSIS] = frequency_analysis['frequency_score']
        
        # 4. Predictability Score
        predictability = self._calculate_predictability_score(pattern_labels, pattern_intensity)
        validation_metrics[ValidationMetric.PREDICTABILITY_SCORE] = predictability
        
        # 5. Temporal Stability
        temporal_analysis = self._analyze_temporal_stability(pattern_labels)
        validation_metrics[ValidationMetric.TEMPORAL_STABILITY] = temporal_analysis['stability_score']
        
        # 6. Magnitude Consistency
        magnitude_analysis = self._analyze_magnitude_consistency(pattern_labels, price_data)
        validation_metrics[ValidationMetric.MAGNITUDE_CONSISTENCY] = magnitude_analysis['consistency_score']
        
        # 7. Duration Distribution
        duration_analysis = self._analyze_duration_distribution(pattern_labels)
        validation_metrics[ValidationMetric.DURATION_DISTRIBUTION] = duration_analysis['distribution_score']
        
        # Overall validation decision
        is_valid = self._make_validation_decision(validation_metrics, statistical_tests, economic_tests)
        
        # Generate summary and recommendations
        validation_summary = self._generate_validation_summary(pattern_name, validation_metrics, is_valid)
        recommendations = self._generate_recommendations(validation_metrics, is_valid)
        
        return PatternValidationResult(
            pattern_name=pattern_name,
            validation_metrics=validation_metrics,
            statistical_tests=statistical_tests,
            economic_tests=economic_tests,
            temporal_analysis=temporal_analysis,
            is_valid_pattern=is_valid,
            validation_summary=validation_summary,
            recommendations=recommendations
        )
    
    def validate_multiple_patterns(self, 
                                 patterns: Dict[str, Dict[str, pd.Series]],
                                 price_data: pd.Series) -> Dict[str, PatternValidationResult]:
        """
        Validate multiple patterns simultaneously.
        
        Args:
            patterns: Dictionary of {pattern_name: {'labels': Series, 'intensity': Series}}
            price_data: Price series for validation
            
        Returns:
            Dictionary of validation results for each pattern
        """
        
        self.logger.info(f"🔍 Validating {len(patterns)} patterns")
        
        validation_results = {}
        
        for pattern_name, pattern_data in patterns.items():
            try:
                result = self.validate_pattern(
                    pattern_labels=pattern_data['labels'],
                    pattern_intensity=pattern_data.get('intensity', pd.Series(0, index=pattern_data['labels'].index)),
                    price_data=price_data,
                    pattern_name=pattern_name
                )
                validation_results[pattern_name] = result
                
                if result.is_valid_pattern:
                    self.logger.info(f"   ✅ {pattern_name}: Valid pattern")
                else:
                    self.logger.info(f"   ❌ {pattern_name}: Invalid pattern")
                    
            except Exception as e:
                self.logger.error(f"   ⚠️ {pattern_name}: Validation failed - {e}")
                continue
        
        return validation_results
    
    def _test_statistical_significance(self, pattern_labels: pd.Series, price_data: pd.Series) -> Dict[str, float]:
        """Test statistical significance of pattern."""
        
        # Align data
        common_index = pattern_labels.index.intersection(price_data.index)
        if len(common_index) < 50:
            return {'p_value': 1.0, 't_statistic': 0.0, 'effect_size': 0.0}
        
        aligned_labels = pattern_labels.loc[common_index]
        aligned_prices = price_data.loc[common_index]
        
        returns = aligned_prices.pct_change().fillna(0)
        
        # Pattern vs non-pattern returns
        pattern_returns = returns[aligned_labels == 1]
        non_pattern_returns = returns[aligned_labels == 0]
        
        if len(pattern_returns) < 10 or len(non_pattern_returns) < 10:
            return {'p_value': 1.0, 't_statistic': 0.0, 'effect_size': 0.0}
        
        try:
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(pattern_returns, non_pattern_returns)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(pattern_returns) - 1) * pattern_returns.var() + 
                                (len(non_pattern_returns) - 1) * non_pattern_returns.var()) / 
                               (len(pattern_returns) + len(non_pattern_returns) - 2))
            
            effect_size = abs(pattern_returns.mean() - non_pattern_returns.mean()) / pooled_std
            
            return {
                'p_value': float(p_value),
                't_statistic': float(abs(t_stat)),
                'effect_size': float(effect_size)
            }
        except:
            return {'p_value': 1.0, 't_statistic': 0.0, 'effect_size': 0.0}
    
    def _test_economic_significance(self, pattern_labels: pd.Series, price_data: pd.Series) -> Dict[str, float]:
        """Test economic significance of pattern."""
        
        # Align data
        common_index = pattern_labels.index.intersection(price_data.index)
        if len(common_index) < 50:
            return {'economic_score': 0.0, 'sharpe_improvement': 0.0, 'hit_rate': 0.5}
        
        aligned_labels = pattern_labels.loc[common_index]
        aligned_prices = price_data.loc[common_index]
        
        returns = aligned_prices.pct_change().fillna(0)
        
        # Pattern-based strategy returns
        pattern_returns = returns[aligned_labels == 1]
        
        if len(pattern_returns) < 10:
            return {'economic_score': 0.0, 'sharpe_improvement': 0.0, 'hit_rate': 0.5}
        
        try:
            # Hit rate
            hit_rate = (pattern_returns > 0).mean()
            
            # Sharpe ratio improvement
            pattern_sharpe = pattern_returns.mean() / pattern_returns.std() if pattern_returns.std() > 0 else 0
            baseline_sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
            sharpe_improvement = pattern_sharpe - baseline_sharpe
            
            # Economic score (composite)
            economic_score = min(1.0, max(0.0, 
                (abs(sharpe_improvement) * 0.5) + 
                (abs(hit_rate - 0.5) * 2 * 0.3) + 
                (min(abs(pattern_returns.mean()) * 1000, 1.0) * 0.2)
            ))
            
            return {
                'economic_score': float(economic_score),
                'sharpe_improvement': float(sharpe_improvement),
                'hit_rate': float(hit_rate)
            }
        except:
            return {'economic_score': 0.0, 'sharpe_improvement': 0.0, 'hit_rate': 0.5}
    
    def _analyze_pattern_frequency(self, pattern_labels: pd.Series) -> Dict[str, float]:
        """Analyze pattern frequency characteristics."""
        
        if len(pattern_labels) == 0:
            return {'frequency_score': 0.0, 'frequency': 0.0}
        
        frequency = pattern_labels.sum() / len(pattern_labels)
        
        # Frequency score (optimal around 5-20%)
        if 0.05 <= frequency <= 0.20:
            frequency_score = 1.0
        elif 0.02 <= frequency <= 0.40:
            frequency_score = 0.7
        elif frequency > 0.40:
            frequency_score = 0.3  # Too frequent, likely noise
        else:
            frequency_score = 0.1  # Too rare, likely overfitting
        
        return {
            'frequency_score': float(frequency_score),
            'frequency': float(frequency)
        }
    
    def _calculate_predictability_score(self, pattern_labels: pd.Series, pattern_intensity: pd.Series) -> float:
        """Calculate pattern predictability score."""
        
        if len(pattern_labels) == 0:
            return 0.0
        
        # Entropy-based predictability for binary labels
        frequency = pattern_labels.sum() / len(pattern_labels)
        
        if frequency == 0 or frequency == 1:
            binary_predictability = 0.0
        else:
            entropy = -frequency * np.log2(frequency) - (1 - frequency) * np.log2(1 - frequency)
            binary_predictability = 1.0 - entropy  # Lower entropy = more predictable
        
        # Intensity consistency
        if len(pattern_intensity) > 0 and pattern_intensity.std() > 0:
            intensity_consistency = 1.0 - (pattern_intensity.std() / (pattern_intensity.mean() + 0.001))
            intensity_consistency = max(0.0, min(1.0, intensity_consistency))
        else:
            intensity_consistency = 0.0
        
        # Combined predictability score
        predictability = (binary_predictability * 0.7) + (intensity_consistency * 0.3)
        
        return float(predictability)
    
    def _analyze_temporal_stability(self, pattern_labels: pd.Series) -> Dict[str, float]:
        """Analyze temporal stability of pattern."""
        
        if len(pattern_labels) < 100:
            return {'stability_score': 0.0, 'consistency': 0.0}
        
        # Split into time periods
        n_periods = 4
        period_size = len(pattern_labels) // n_periods
        period_frequencies = []
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(pattern_labels)
            
            period_labels = pattern_labels.iloc[start_idx:end_idx]
            period_frequency = period_labels.sum() / len(period_labels)
            period_frequencies.append(period_frequency)
        
        # Stability = 1 - coefficient of variation
        if len(period_frequencies) > 1:
            mean_freq = np.mean(period_frequencies)
            std_freq = np.std(period_frequencies)
            
            if mean_freq > 0:
                cv = std_freq / mean_freq
                stability_score = max(0.0, 1.0 - cv)
            else:
                stability_score = 0.0
        else:
            stability_score = 0.0
        
        return {
            'stability_score': float(stability_score),
            'consistency': float(1.0 - np.std(period_frequencies)) if period_frequencies else 0.0
        }
    
    def _analyze_magnitude_consistency(self, pattern_labels: pd.Series, price_data: pd.Series) -> Dict[str, float]:
        """Analyze magnitude consistency of pattern."""
        
        # Align data
        common_index = pattern_labels.index.intersection(price_data.index)
        if len(common_index) < 20:
            return {'consistency_score': 0.0, 'magnitude_cv': 1.0}
        
        aligned_labels = pattern_labels.loc[common_index]
        aligned_prices = price_data.loc[common_index]
        
        returns = aligned_prices.pct_change().fillna(0)
        pattern_returns = returns[aligned_labels == 1]
        
        if len(pattern_returns) < 10:
            return {'consistency_score': 0.0, 'magnitude_cv': 1.0}
        
        # Coefficient of variation for magnitudes
        abs_returns = abs(pattern_returns)
        if abs_returns.mean() > 0:
            magnitude_cv = abs_returns.std() / abs_returns.mean()
            consistency_score = max(0.0, 1.0 - magnitude_cv)
        else:
            magnitude_cv = 1.0
            consistency_score = 0.0
        
        return {
            'consistency_score': float(consistency_score),
            'magnitude_cv': float(magnitude_cv)
        }
    
    def _analyze_duration_distribution(self, pattern_labels: pd.Series) -> Dict[str, float]:
        """Analyze duration distribution of pattern."""
        
        # Calculate pattern durations
        durations = []
        current_duration = 0
        in_pattern = False
        
        for label in pattern_labels:
            if label == 1:
                if not in_pattern:
                    in_pattern = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_pattern:
                    durations.append(current_duration)
                    in_pattern = False
                    current_duration = 0
        
        if in_pattern and current_duration > 0:
            durations.append(current_duration)
        
        if not durations:
            return {'distribution_score': 0.0, 'mean_duration': 0.0}
        
        # Distribution quality score
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        # Prefer patterns with reasonable duration (2-10 periods) and low variance
        if 2 <= mean_duration <= 10:
            duration_score = 1.0
        elif 1 <= mean_duration <= 20:
            duration_score = 0.7
        else:
            duration_score = 0.3
        
        # Penalize high variance
        if std_duration > 0:
            cv = std_duration / mean_duration
            variance_penalty = max(0.0, 1.0 - cv)
        else:
            variance_penalty = 1.0
        
        distribution_score = duration_score * variance_penalty
        
        return {
            'distribution_score': float(distribution_score),
            'mean_duration': float(mean_duration)
        }
    
    def _make_validation_decision(self, 
                                validation_metrics: Dict[ValidationMetric, float],
                                statistical_tests: Dict[str, float],
                                economic_tests: Dict[str, float]) -> bool:
        """Make overall validation decision."""
        
        # Minimum requirements
        min_requirements = [
            validation_metrics.get(ValidationMetric.STATISTICAL_SIGNIFICANCE, 0) > 0.95,  # p < 0.05
            validation_metrics.get(ValidationMetric.FREQUENCY_ANALYSIS, 0) > 0.5,
            statistical_tests.get('effect_size', 0) > 0.2  # Minimum effect size
        ]
        
        # Additional quality criteria
        quality_criteria = [
            validation_metrics.get(ValidationMetric.ECONOMIC_SIGNIFICANCE, 0) > 0.3,
            validation_metrics.get(ValidationMetric.PREDICTABILITY_SCORE, 0) > 0.2,
            validation_metrics.get(ValidationMetric.TEMPORAL_STABILITY, 0) > 0.3,
            validation_metrics.get(ValidationMetric.MAGNITUDE_CONSISTENCY, 0) > 0.3
        ]
        
        # Pattern is valid if it meets minimum requirements AND at least 2 quality criteria
        meets_minimums = all(min_requirements)
        quality_score = sum(quality_criteria)
        
        return meets_minimums and quality_score >= 2
    
    def _generate_validation_summary(self, 
                                   pattern_name: str,
                                   validation_metrics: Dict[ValidationMetric, float],
                                   is_valid: bool) -> str:
        """Generate validation summary text."""
        
        status = "VALID" if is_valid else "INVALID"
        
        summary = f"Pattern '{pattern_name}' validation: {status}\n\n"
        summary += "Validation Metrics:\n"
        
        for metric, value in validation_metrics.items():
            summary += f"  - {metric.value.replace('_', ' ').title()}: {value:.3f}\n"
        
        if is_valid:
            summary += "\nPattern meets validation criteria and is suitable for trading analysis."
        else:
            summary += "\nPattern does not meet validation criteria. Consider refinement or exclusion."
        
        return summary
    
    def _generate_recommendations(self, 
                                validation_metrics: Dict[ValidationMetric, float],
                                is_valid: bool) -> List[str]:
        """Generate specific recommendations based on validation results."""
        
        recommendations = []
        
        if is_valid:
            recommendations.append("✅ Pattern is valid for trading analysis")
            
            # Specific strengths
            if validation_metrics.get(ValidationMetric.ECONOMIC_SIGNIFICANCE, 0) > 0.7:
                recommendations.append("🎯 Strong economic significance - prioritize for strategy development")
            
            if validation_metrics.get(ValidationMetric.TEMPORAL_STABILITY, 0) > 0.7:
                recommendations.append("📈 High temporal stability - suitable for long-term strategies")
            
            if validation_metrics.get(ValidationMetric.PREDICTABILITY_SCORE, 0) > 0.7:
                recommendations.append("🔮 High predictability - good for ML model training")
        
        else:
            recommendations.append("❌ Pattern validation failed")
            
            # Specific issues
            if validation_metrics.get(ValidationMetric.STATISTICAL_SIGNIFICANCE, 0) < 0.95:
                recommendations.append("📊 Improve statistical significance (p-value too high)")
            
            if validation_metrics.get(ValidationMetric.FREQUENCY_ANALYSIS, 0) < 0.5:
                recommendations.append("📉 Pattern frequency issues - too rare or too common")
            
            if validation_metrics.get(ValidationMetric.ECONOMIC_SIGNIFICANCE, 0) < 0.3:
                recommendations.append("💰 Low economic value - may not be profitable")
            
            if validation_metrics.get(ValidationMetric.TEMPORAL_STABILITY, 0) < 0.3:
                recommendations.append("⏰ Poor temporal stability - pattern may be regime-dependent")
        
        return recommendations