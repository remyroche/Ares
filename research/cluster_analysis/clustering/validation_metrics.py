"""
Regime Validation and Quality Metrics System.

This module provides comprehensive validation and quality assessment metrics
specifically designed for market regime clustering and identification. It
implements various metrics to evaluate the quality, stability, and trading
relevance of discovered regimes.

Key Validation Areas:
- Clustering Quality (silhouette, calinski-harabasz, davies-bouldin)
- Regime Stability (temporal consistency, persistence)
- Economic Significance (return differences, volatility regimes)
- Trading Relevance (regime-specific performance, transition costs)
- Statistical Validation (regime separation, homogeneity)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
from scipy import stats
from abc import ABC, abstractmethod

from src.utils.logger import system_logger

# Import economic metrics
try:
    from .economic_metrics import EconomicValidator, EconomicValidationConfig, EconomicMetric
    ECONOMIC_METRICS_AVAILABLE = True
except ImportError:
    ECONOMIC_METRICS_AVAILABLE = False


class ValidationMetric(Enum):
    """Enumeration of validation metrics."""
    # Clustering quality metrics
    SILHOUETTE_SCORE = "silhouette_score"
    CALINSKI_HARABASZ = "calinski_harabasz"
    DAVIES_BOULDIN = "davies_bouldin"
    ADJUSTED_RAND_INDEX = "adjusted_rand_index"
    
    # Regime stability metrics
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    REGIME_PERSISTENCE = "regime_persistence"
    TRANSITION_FREQUENCY = "transition_frequency"
    REGIME_DURATION = "regime_duration"
    
    # Economic significance metrics
    RETURN_SEPARABILITY = "return_separability"
    VOLATILITY_SEPARABILITY = "volatility_separability"
    SHARPE_RATIO_DIFFERENCE = "sharpe_ratio_difference"
    MAXIMUM_DRAWDOWN_DIFFERENCE = "maximum_drawdown_difference"
    
    # Trading relevance metrics
    REGIME_PREDICTABILITY = "regime_predictability"
    TRADING_SIGNAL_QUALITY = "trading_signal_quality"
    REGIME_TRANSITION_COST = "regime_transition_cost"
    PORTFOLIO_PERFORMANCE = "portfolio_performance"
    
    # Statistical validation metrics
    REGIME_HOMOGENEITY = "regime_homogeneity"
    INTER_REGIME_SEPARATION = "inter_regime_separation"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    REGIME_BALANCE = "regime_balance"


@dataclass
class ValidationConfig:
    """Configuration for regime validation."""
    # General parameters
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # Stability parameters
    min_regime_duration: int = 10
    transition_threshold: float = 0.1
    
    # Economic parameters
    risk_free_rate: float = 0.02
    transaction_cost: float = 0.001
    
    # Trading parameters
    lookback_period: int = 252
    rebalancing_frequency: int = 21
    
    # Statistical parameters
    normality_test: str = "shapiro"  # shapiro, kstest, jarque_bera
    homogeneity_test: str = "levene"  # levene, bartlett
    
    # Performance parameters
    benchmark_return: float = 0.08
    max_acceptable_drawdown: float = 0.20


@dataclass
class ValidationResult:
    """Result container for validation metrics."""
    metric: ValidationMetric
    value: float
    confidence_interval: Optional[Tuple[float, float]]
    p_value: Optional[float]
    interpretation: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric': self.metric.value,
            'value': self.value,
            'confidence_interval': self.confidence_interval,
            'p_value': self.p_value,
            'interpretation': self.interpretation,
            'metadata': self.metadata
        }
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if the metric is statistically significant."""
        return self.p_value is not None and self.p_value < alpha


class BaseValidator(ABC):
    """Abstract base class for validation metrics."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = system_logger.getChild(f'Validator.{self.__class__.__name__}')
    
    @abstractmethod
    def validate(self, 
                data: pd.DataFrame,
                regime_labels: np.ndarray,
                **kwargs) -> ValidationResult:
        """Validate regime quality using this metric."""
        pass
    
    def _bootstrap_confidence_interval(self, 
                                     data: np.ndarray,
                                     statistic_func,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        bootstrap_stats = []
        
        for _ in range(self.config.bootstrap_samples):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)
        
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return (lower, upper)


class SilhouetteValidator(BaseValidator):
    """Silhouette score validation."""
    
    def validate(self, data: pd.DataFrame, regime_labels: np.ndarray, **kwargs) -> ValidationResult:
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.fillna(0))
        
        # Calculate silhouette score
        score = silhouette_score(data_scaled, regime_labels)
        
        # Interpretation
        if score > 0.7:
            interpretation = "Excellent cluster separation"
        elif score > 0.5:
            interpretation = "Good cluster separation"
        elif score > 0.25:
            interpretation = "Weak cluster separation"
        else:
            interpretation = "Poor cluster separation"
        
        return ValidationResult(
            metric=ValidationMetric.SILHOUETTE_SCORE,
            value=float(score),
            confidence_interval=None,  # Could add bootstrap CI
            p_value=None,
            interpretation=interpretation,
            metadata={
                'n_clusters': len(np.unique(regime_labels)),
                'n_samples': len(data)
            }
        )


class TemporalConsistencyValidator(BaseValidator):
    """Temporal consistency validation."""
    
    def validate(self, data: pd.DataFrame, regime_labels: np.ndarray, **kwargs) -> ValidationResult:
        # Calculate regime persistence
        regime_changes = np.diff(regime_labels) != 0
        n_changes = np.sum(regime_changes)
        total_periods = len(regime_labels) - 1
        
        # Consistency score (1 - change rate)
        consistency_score = 1 - (n_changes / total_periods)
        
        # Calculate average regime duration
        regime_durations = []
        current_regime = regime_labels[0]
        current_duration = 1
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] == current_regime:
                current_duration += 1
            else:
                regime_durations.append(current_duration)
                current_regime = regime_labels[i]
                current_duration = 1
        
        regime_durations.append(current_duration)
        avg_duration = np.mean(regime_durations)
        
        # Interpretation
        if consistency_score > 0.8:
            interpretation = "High temporal consistency"
        elif consistency_score > 0.6:
            interpretation = "Moderate temporal consistency"
        else:
            interpretation = "Low temporal consistency"
        
        return ValidationResult(
            metric=ValidationMetric.TEMPORAL_CONSISTENCY,
            value=float(consistency_score),
            confidence_interval=None,
            p_value=None,
            interpretation=interpretation,
            metadata={
                'n_transitions': int(n_changes),
                'avg_regime_duration': float(avg_duration),
                'regime_durations': regime_durations
            }
        )


class ReturnSeparabilityValidator(BaseValidator):
    """Return separability validation."""
    
    def validate(self, data: pd.DataFrame, regime_labels: np.ndarray, **kwargs) -> ValidationResult:
        # Assume 'returns' column exists or calculate from 'close'
        if 'returns' in data.columns:
            returns = data['returns']
        elif 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            regime_labels = regime_labels[1:]  # Align with returns
        else:
            raise ValueError("No returns or close price data available")
        
        # Group returns by regime
        unique_regimes = np.unique(regime_labels)
        regime_returns = {}
        
        for regime in unique_regimes:
            mask = regime_labels == regime
            regime_returns[regime] = returns[mask].dropna()
        
        # Calculate ANOVA F-statistic
        regime_return_lists = list(regime_returns.values())
        if len(regime_return_lists) < 2:
            f_stat, p_value = 0.0, 1.0
        else:
            f_stat, p_value = stats.f_oneway(*regime_return_lists)
        
        # Calculate effect size (eta-squared)
        if len(regime_return_lists) >= 2:
            ss_between = sum(len(group) * (np.mean(group) - np.mean(returns)) ** 2 
                           for group in regime_return_lists)
            ss_total = np.sum((returns - np.mean(returns)) ** 2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
        else:
            eta_squared = 0
        
        # Interpretation
        if p_value < 0.001:
            interpretation = "Highly significant return differences between regimes"
        elif p_value < 0.05:
            interpretation = "Significant return differences between regimes"
        else:
            interpretation = "No significant return differences between regimes"
        
        return ValidationResult(
            metric=ValidationMetric.RETURN_SEPARABILITY,
            value=float(eta_squared),
            confidence_interval=None,
            p_value=float(p_value),
            interpretation=interpretation,
            metadata={
                'f_statistic': float(f_stat),
                'regime_mean_returns': {int(k): float(np.mean(v)) for k, v in regime_returns.items()},
                'regime_std_returns': {int(k): float(np.std(v)) for k, v in regime_returns.items()}
            }
        )


class VolatilitySeparabilityValidator(BaseValidator):
    """Volatility separability validation."""
    
    def validate(self, data: pd.DataFrame, regime_labels: np.ndarray, **kwargs) -> ValidationResult:
        # Calculate returns if not available
        if 'returns' in data.columns:
            returns = data['returns']
        elif 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            regime_labels = regime_labels[1:]
        else:
            raise ValueError("No returns or close price data available")
        
        # Calculate rolling volatility
        volatility_window = kwargs.get('volatility_window', 20)
        volatility = returns.rolling(volatility_window).std() * np.sqrt(252)  # Annualized
        volatility = volatility.dropna()
        
        # Align regime labels with volatility
        regime_labels_aligned = regime_labels[volatility_window-1:]
        
        # Group volatility by regime
        unique_regimes = np.unique(regime_labels_aligned)
        regime_volatilities = {}
        
        for regime in unique_regimes:
            mask = regime_labels_aligned == regime
            regime_volatilities[regime] = volatility[mask].dropna()
        
        # Calculate ANOVA F-statistic for volatility
        regime_vol_lists = list(regime_volatilities.values())
        if len(regime_vol_lists) < 2:
            f_stat, p_value = 0.0, 1.0
        else:
            f_stat, p_value = stats.f_oneway(*regime_vol_lists)
        
        # Calculate effect size
        if len(regime_vol_lists) >= 2:
            ss_between = sum(len(group) * (np.mean(group) - np.mean(volatility)) ** 2 
                           for group in regime_vol_lists)
            ss_total = np.sum((volatility - np.mean(volatility)) ** 2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
        else:
            eta_squared = 0
        
        # Interpretation
        if p_value < 0.001:
            interpretation = "Highly significant volatility differences between regimes"
        elif p_value < 0.05:
            interpretation = "Significant volatility differences between regimes"
        else:
            interpretation = "No significant volatility differences between regimes"
        
        return ValidationResult(
            metric=ValidationMetric.VOLATILITY_SEPARABILITY,
            value=float(eta_squared),
            confidence_interval=None,
            p_value=float(p_value),
            interpretation=interpretation,
            metadata={
                'f_statistic': float(f_stat),
                'volatility_window': volatility_window,
                'regime_mean_volatility': {int(k): float(np.mean(v)) for k, v in regime_volatilities.items()},
                'regime_std_volatility': {int(k): float(np.std(v)) for k, v in regime_volatilities.items()}
            }
        )


class RegimePredictabilityValidator(BaseValidator):
    """Regime predictability validation."""
    
    def validate(self, data: pd.DataFrame, regime_labels: np.ndarray, **kwargs) -> ValidationResult:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        # Prepare features (lagged regime labels and market features)
        n_lags = kwargs.get('n_lags', 5)
        features = []
        targets = []
        
        # Create lagged features
        for i in range(n_lags, len(regime_labels) - 1):
            # Lagged regime labels
            lagged_regimes = regime_labels[i-n_lags:i]
            
            # Market features at time i
            if 'close' in data.columns:
                market_features = [
                    data['close'].iloc[i] / data['close'].iloc[i-1] - 1,  # Return
                    data['close'].iloc[i-n_lags:i].std() / data['close'].iloc[i],  # Volatility proxy
                ]
            else:
                market_features = []
            
            # Combine features
            feature_vector = list(lagged_regimes) + market_features
            features.append(feature_vector)
            targets.append(regime_labels[i + 1])  # Predict next regime
        
        if not features:
            return ValidationResult(
                metric=ValidationMetric.REGIME_PREDICTABILITY,
                value=0.0,
                confidence_interval=None,
                p_value=None,
                interpretation="Insufficient data for predictability analysis",
                metadata={'n_samples': 0}
            )
        
        X = np.array(features)
        y = np.array(targets)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train classifier and get cross-validation scores
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
        
        predictability_score = np.mean(cv_scores)
        
        # Calculate baseline accuracy (most frequent class)
        baseline_accuracy = np.max(np.bincount(y)) / len(y)
        
        # Interpretation
        improvement = predictability_score - baseline_accuracy
        if improvement > 0.1:
            interpretation = "Regimes are highly predictable"
        elif improvement > 0.05:
            interpretation = "Regimes are moderately predictable"
        else:
            interpretation = "Regimes are poorly predictable"
        
        return ValidationResult(
            metric=ValidationMetric.REGIME_PREDICTABILITY,
            value=float(predictability_score),
            confidence_interval=None,
            p_value=None,
            interpretation=interpretation,
            metadata={
                'baseline_accuracy': float(baseline_accuracy),
                'improvement_over_baseline': float(improvement),
                'cv_scores': cv_scores.tolist(),
                'n_samples': len(features),
                'n_lags': n_lags
            }
        )


class RegimeBalanceValidator(BaseValidator):
    """Regime balance validation."""
    
    def validate(self, data: pd.DataFrame, regime_labels: np.ndarray, **kwargs) -> ValidationResult:
        # Calculate regime distribution
        unique_regimes, counts = np.unique(regime_labels, return_counts=True)
        proportions = counts / len(regime_labels)
        
        # Calculate balance metrics
        # 1. Entropy (higher is more balanced)
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        max_entropy = np.log2(len(unique_regimes))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # 2. Gini coefficient (lower is more balanced)
        gini = 1 - np.sum(proportions ** 2)
        max_gini = 1 - 1/len(unique_regimes)
        normalized_gini = gini / max_gini if max_gini > 0 else 0
        
        # 3. Coefficient of variation
        cv = np.std(proportions) / np.mean(proportions)
        
        # Use normalized entropy as main balance score
        balance_score = normalized_entropy
        
        # Interpretation
        if balance_score > 0.9:
            interpretation = "Highly balanced regime distribution"
        elif balance_score > 0.7:
            interpretation = "Moderately balanced regime distribution"
        else:
            interpretation = "Imbalanced regime distribution"
        
        return ValidationResult(
            metric=ValidationMetric.REGIME_BALANCE,
            value=float(balance_score),
            confidence_interval=None,
            p_value=None,
            interpretation=interpretation,
            metadata={
                'regime_counts': dict(zip(unique_regimes.astype(int), counts.astype(int))),
                'regime_proportions': dict(zip(unique_regimes.astype(int), proportions.astype(float))),
                'entropy': float(entropy),
                'gini_coefficient': float(gini),
                'coefficient_of_variation': float(cv),
                'n_regimes': len(unique_regimes)
            }
        )


class RegimeHomogeneityValidator(BaseValidator):
    """Regime homogeneity validation."""
    
    def validate(self, data: pd.DataFrame, regime_labels: np.ndarray, **kwargs) -> ValidationResult:
        # Calculate within-regime variance for each feature
        unique_regimes = np.unique(regime_labels)
        feature_homogeneity_scores = []
        
        for col in data.select_dtypes(include=[np.number]).columns:
            feature_data = data[col].fillna(data[col].mean())
            
            # Calculate within-regime variances
            within_regime_vars = []
            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                regime_data = feature_data[regime_mask]
                if len(regime_data) > 1:
                    within_regime_vars.append(np.var(regime_data))
                else:
                    within_regime_vars.append(0)
            
            # Calculate total variance
            total_var = np.var(feature_data)
            
            # Homogeneity score (1 - within_variance/total_variance)
            avg_within_var = np.mean(within_regime_vars)
            homogeneity = 1 - (avg_within_var / total_var) if total_var > 0 else 1
            feature_homogeneity_scores.append(homogeneity)
        
        # Overall homogeneity score
        overall_homogeneity = np.mean(feature_homogeneity_scores)
        
        # Interpretation
        if overall_homogeneity > 0.7:
            interpretation = "High regime homogeneity"
        elif overall_homogeneity > 0.5:
            interpretation = "Moderate regime homogeneity"
        else:
            interpretation = "Low regime homogeneity"
        
        return ValidationResult(
            metric=ValidationMetric.REGIME_HOMOGENEITY,
            value=float(overall_homogeneity),
            confidence_interval=None,
            p_value=None,
            interpretation=interpretation,
            metadata={
                'feature_homogeneity_scores': feature_homogeneity_scores,
                'n_features': len(feature_homogeneity_scores),
                'n_regimes': len(unique_regimes)
            }
        )


class RegimeValidationMetrics:
    """
    Main regime validation and quality metrics system.
    
    This class provides comprehensive validation and assessment of market
    regime clustering quality, stability, and trading relevance.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the regime validation system.
        
        Args:
            config: Configuration for validation metrics
        """
        self.config = config or ValidationConfig()
        self.logger = system_logger.getChild('RegimeValidationMetrics')
        self.results: Dict[ValidationMetric, ValidationResult] = {}
        
        # Initialize validators
        self.validators = {
            ValidationMetric.SILHOUETTE_SCORE: SilhouetteValidator(self.config),
            ValidationMetric.TEMPORAL_CONSISTENCY: TemporalConsistencyValidator(self.config),
            ValidationMetric.RETURN_SEPARABILITY: ReturnSeparabilityValidator(self.config),
            ValidationMetric.VOLATILITY_SEPARABILITY: VolatilitySeparabilityValidator(self.config),
            ValidationMetric.REGIME_PREDICTABILITY: RegimePredictabilityValidator(self.config),
            ValidationMetric.REGIME_BALANCE: RegimeBalanceValidator(self.config),
            ValidationMetric.REGIME_HOMOGENEITY: RegimeHomogeneityValidator(self.config)
        }
        
        # Initialize economic validator if available
        if ECONOMIC_METRICS_AVAILABLE:
            self.economic_validator = EconomicValidator(EconomicValidationConfig())
        else:
            self.economic_validator = None
    
    def validate_single_metric(self,
                              data: pd.DataFrame,
                              regime_labels: np.ndarray,
                              metric: ValidationMetric,
                              **kwargs) -> ValidationResult:
        """
        Validate using a single metric.
        
        Args:
            data: Market data
            regime_labels: Regime assignments
            metric: Validation metric to use
            **kwargs: Additional parameters for specific metrics
            
        Returns:
            Validation result
        """
        self.logger.info(f"🔍 Validating using {metric.value}")
        
        if metric not in self.validators:
            raise ValueError(f"Validation metric {metric.value} not supported")
        
        result = self.validators[metric].validate(data, regime_labels, **kwargs)
        self.results[metric] = result
        
        self.logger.info(f"✅ {metric.value}: {result.value:.3f} - {result.interpretation}")
        
        return result
    
    def validate_all_metrics(self,
                           data: pd.DataFrame,
                           regime_labels: np.ndarray,
                           **kwargs) -> Dict[ValidationMetric, ValidationResult]:
        """
        Validate using all available metrics.
        
        Args:
            data: Market data
            regime_labels: Regime assignments
            **kwargs: Additional parameters for specific metrics
            
        Returns:
            Dictionary mapping metrics to results
        """
        self.logger.info("🚀 Running comprehensive regime validation")
        
        results = {}
        
        for metric in ValidationMetric:
            if metric in self.validators:
                try:
                    result = self.validate_single_metric(data, regime_labels, metric, **kwargs)
                    results[metric] = result
                except Exception as e:
                    self.logger.error(f"❌ {metric.value} failed: {e}")
                    continue
        
        self.logger.info(f"✅ Completed {len(results)} validation metrics")
        return results
    
    def validate_economic_significance(self,
                                     market_data: pd.DataFrame,
                                     regime_labels: np.ndarray) -> Dict[str, Any]:
        """
        Validate economic significance of discovered regimes.
        
        Args:
            market_data: Market data
            regime_labels: Regime assignments
            
        Returns:
            Dictionary with economic validation results
        """
        if not ECONOMIC_METRICS_AVAILABLE or self.economic_validator is None:
            self.logger.warning("Economic metrics not available")
            return {}
        
        self.logger.info("💰 Running economic significance validation")
        
        try:
            economic_results = self.economic_validator.validate_regime_economics(market_data, regime_labels)
            
            # Convert to serializable format
            economic_dict = {
                metric.value: result.to_dict() 
                for metric, result in economic_results.items()
            }
            
            # Generate economic report
            economic_report = self.economic_validator.generate_economic_report(economic_results)
            
            # Calculate economic significance summary
            economically_significant = sum(1 for result in economic_results.values() if result.economic_significance)
            total_metrics = len(economic_results)
            economic_significance_rate = economically_significant / total_metrics if total_metrics > 0 else 0
            
            summary = {
                'total_economic_metrics': total_metrics,
                'economically_significant_metrics': economically_significant,
                'economic_significance_rate': economic_significance_rate,
                'overall_economic_quality': 'strong' if economic_significance_rate >= 0.7 else 'moderate' if economic_significance_rate >= 0.4 else 'weak'
            }
            
            self.logger.info(f"💰 Economic validation completed: {economically_significant}/{total_metrics} metrics significant")
            
            return {
                'economic_results': economic_dict,
                'economic_report': economic_report,
                'economic_summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"Economic validation failed: {e}")
            return {'error': str(e)}
    
    def calculate_composite_score(self, 
                                weights: Optional[Dict[ValidationMetric, float]] = None) -> float:
        """
        Calculate composite validation score.
        
        Args:
            weights: Custom weights for metrics (default: equal weights)
            
        Returns:
            Composite validation score
        """
        if not self.results:
            return 0.0
        
        if weights is None:
            weights = {metric: 1.0 for metric in self.results.keys()}
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, result in self.results.items():
            weight = weights.get(metric, 0.0)
            if weight > 0:
                weighted_sum += weight * result.value
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of validation results.
        
        Returns:
            Dictionary with validation summary
        """
        if not self.results:
            return {}
        
        # Categorize metrics
        quality_metrics = [ValidationMetric.SILHOUETTE_SCORE, ValidationMetric.REGIME_HOMOGENEITY]
        stability_metrics = [ValidationMetric.TEMPORAL_CONSISTENCY, ValidationMetric.REGIME_BALANCE]
        economic_metrics = [ValidationMetric.RETURN_SEPARABILITY, ValidationMetric.VOLATILITY_SEPARABILITY]
        predictability_metrics = [ValidationMetric.REGIME_PREDICTABILITY]
        
        summary = {
            'overall_score': self.calculate_composite_score(),
            'n_metrics': len(self.results),
            'significant_metrics': sum(1 for r in self.results.values() if r.is_significant()),
            'category_scores': {}
        }
        
        # Calculate category scores
        for category, metrics in [
            ('quality', quality_metrics),
            ('stability', stability_metrics),
            ('economic', economic_metrics),
            ('predictability', predictability_metrics)
        ]:
            category_results = [self.results[m] for m in metrics if m in self.results]
            if category_results:
                category_score = np.mean([r.value for r in category_results])
                summary['category_scores'][category] = float(category_score)
        
        return summary
    
    def compare_regime_sets(self,
                          data: pd.DataFrame,
                          regime_sets: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Compare multiple regime clustering results.
        
        Args:
            data: Market data
            regime_sets: Dictionary mapping names to regime label arrays
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for name, regime_labels in regime_sets.items():
            self.logger.info(f"🔍 Validating regime set: {name}")
            
            # Run validation
            results = self.validate_all_metrics(data, regime_labels)
            
            # Create comparison row
            row = {'regime_set': name}
            for metric, result in results.items():
                row[metric.value] = result.value
                if result.p_value is not None:
                    row[f"{metric.value}_pvalue"] = result.p_value
            
            # Add composite score
            row['composite_score'] = self.calculate_composite_score()
            
            comparison_results.append(row)
        
        df = pd.DataFrame(comparison_results)
        
        # Add rankings
        if len(df) > 1:
            df['composite_rank'] = df['composite_score'].rank(ascending=False)
        
        return df.sort_values('composite_score', ascending=False) if not df.empty else df
    
    def save_results(self, filepath: str):
        """Save validation results to file."""
        results_dict = {
            metric.value: result.to_dict() 
            for metric, result in self.results.items()
        }
        
        # Add summary
        results_dict['summary'] = self.get_validation_summary()
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"💾 Saved validation results to {filepath}")
    
    def load_results(self, filepath: str):
        """Load validation results from file."""
        with open(filepath, 'r') as f:
            results_dict = json.load(f)
        
        self.results = {}
        for metric_name, result_dict in results_dict.items():
            if metric_name == 'summary':
                continue
                
            metric = ValidationMetric(metric_name)
            
            # Reconstruct ValidationResult
            result = ValidationResult(
                metric=metric,
                value=result_dict['value'],
                confidence_interval=tuple(result_dict['confidence_interval']) if result_dict['confidence_interval'] else None,
                p_value=result_dict['p_value'],
                interpretation=result_dict['interpretation'],
                metadata=result_dict['metadata']
            )
            
            self.results[metric] = result
        
        self.logger.info(f"📂 Loaded validation results from {filepath}")
    
    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        if not self.results:
            return "No validation results available. Run validation first."
        
        report = []
        report.append("# Regime Validation Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        summary = self.get_validation_summary()
        report.append("## Validation Summary")
        report.append("")
        report.append(f"- **Overall Score**: {summary.get('overall_score', 0):.3f}")
        report.append(f"- **Metrics Evaluated**: {summary.get('n_metrics', 0)}")
        report.append(f"- **Significant Metrics**: {summary.get('significant_metrics', 0)}")
        report.append("")
        
        # Category scores
        if 'category_scores' in summary:
            report.append("**Category Scores:**")
            for category, score in summary['category_scores'].items():
                report.append(f"- {category.title()}: {score:.3f}")
            report.append("")
        
        # Detailed results
        report.append("## Detailed Validation Results")
        report.append("")
        
        # Group by category
        categories = {
            'Clustering Quality': [ValidationMetric.SILHOUETTE_SCORE, ValidationMetric.REGIME_HOMOGENEITY],
            'Temporal Stability': [ValidationMetric.TEMPORAL_CONSISTENCY, ValidationMetric.REGIME_BALANCE],
            'Economic Significance': [ValidationMetric.RETURN_SEPARABILITY, ValidationMetric.VOLATILITY_SEPARABILITY],
            'Predictability': [ValidationMetric.REGIME_PREDICTABILITY]
        }
        
        for category_name, metrics in categories.items():
            category_results = [(m, self.results[m]) for m in metrics if m in self.results]
            
            if category_results:
                report.append(f"### {category_name}")
                report.append("")
                
                for metric, result in category_results:
                    report.append(f"**{metric.value.replace('_', ' ').title()}**")
                    report.append(f"- Value: {result.value:.3f}")
                    if result.p_value is not None:
                        report.append(f"- P-value: {result.p_value:.3f}")
                        significance = "Significant" if result.is_significant() else "Not significant"
                        report.append(f"- Statistical Significance: {significance}")
                    report.append(f"- Interpretation: {result.interpretation}")
                    
                    # Key metadata
                    if result.metadata:
                        report.append("- Key Details:")
                        for key, value in list(result.metadata.items())[:3]:  # Top 3 details
                            if isinstance(value, (int, float)):
                                report.append(f"  - {key.replace('_', ' ').title()}: {value}")
                            elif isinstance(value, str):
                                report.append(f"  - {key.replace('_', ' ').title()}: {value}")
                    
                    report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        overall_score = summary.get('overall_score', 0)
        if overall_score > 0.7:
            report.append("✅ **Excellent regime quality** - Regimes are well-separated, stable, and economically meaningful.")
        elif overall_score > 0.5:
            report.append("⚠️ **Good regime quality** - Regimes show reasonable separation but may need refinement.")
        else:
            report.append("❌ **Poor regime quality** - Consider different clustering methods or feature engineering.")
        
        report.append("")
        
        # Specific recommendations based on low scores
        if ValidationMetric.SILHOUETTE_SCORE in self.results:
            silhouette_score = self.results[ValidationMetric.SILHOUETTE_SCORE].value
            if silhouette_score < 0.3:
                report.append("- Consider increasing or decreasing the number of clusters")
                report.append("- Try different clustering algorithms (e.g., DBSCAN, GMM)")
        
        if ValidationMetric.TEMPORAL_CONSISTENCY in self.results:
            consistency_score = self.results[ValidationMetric.TEMPORAL_CONSISTENCY].value
            if consistency_score < 0.5:
                report.append("- Add temporal smoothing to regime assignments")
                report.append("- Consider minimum regime duration constraints")
        
        if ValidationMetric.RETURN_SEPARABILITY in self.results:
            separability = self.results[ValidationMetric.RETURN_SEPARABILITY]
            if not separability.is_significant():
                report.append("- Regimes may not be economically meaningful")
                report.append("- Consider different features or time periods")
        
        return "\n".join(report)