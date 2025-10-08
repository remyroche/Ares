"""
Quantitative Soundness Validation for Pre-Training Pipeline.

This module implements rigorous statistical and economic validation checks to ensure
the scientific soundness of the pre-training pipeline outputs.

Validation Tests:
1. Label autocorrelation decay - Ensure labels are not trivially predictable
2. Feature-target mutual information - Filter out noise features
3. Feature stability across regimes - Robustness check
4. Sharpe of synthetic signal - Economic plausibility
5. Lookback sensitivity - Robustness of optimal window
6. IC (Information Coefficient) mean/volatility - Predictive quality

Based on quantitative finance best practices from:
- Advances in Financial Machine Learning (López de Prado, 2018)
- Quantitative Equity Portfolio Management (Chincarini & Kim, 2006)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp

from src.utils.logger import system_logger


class ValidationStatus(Enum):
    """Status of a validation check."""
    
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    
    test_name: str
    status: ValidationStatus
    value: float
    threshold: float
    passed: bool
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'status': self.status.value,
            'value': self.value,
            'threshold': self.threshold,
            'passed': self.passed,
            'message': self.message,
            'metadata': self.metadata
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    
    results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed(self) -> bool:
        """Whether all critical checks passed."""
        return all(r.passed or r.status == ValidationStatus.SKIPPED for r in self.results)
    
    @property
    def warnings_count(self) -> int:
        """Number of warnings."""
        return sum(1 for r in self.results if r.status == ValidationStatus.WARNING)
    
    @property
    def failures_count(self) -> int:
        """Number of failures."""
        return sum(1 for r in self.results if r.status == ValidationStatus.FAILED)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'passed': self.passed,
            'warnings_count': self.warnings_count,
            'failures_count': self.failures_count,
            'results': [r.to_dict() for r in self.results],
            'summary': self.summary,
            'metadata': self.metadata
        }


class QuantitativeValidator:
    """
    Implements quantitative validation checks for pre-training outputs.
    
    This validator performs statistical and economic tests to ensure:
    - Labels are economically meaningful and not trivially predictable
    - Features have genuine predictive power
    - Models are robust across different market regimes
    - Optimal parameters (lookbacks) are stable under resampling
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        strict_mode: bool = False
    ):
        """
        Initialize the validator.
        
        Args:
            logger: Optional logger instance
            strict_mode: If True, warnings are treated as failures
        """
        self.logger = logger or system_logger.getChild('QuantitativeValidator')
        self.strict_mode = strict_mode
        self.report = ValidationReport()
    
    def validate_all(
        self,
        labels: Optional[pd.DataFrame] = None,
        features: Optional[pd.DataFrame] = None,
        lookback_results: Optional[Dict[str, Any]] = None,
        regime_labels: Optional[pd.Series] = None
    ) -> ValidationReport:
        """
        Run all validation checks.
        
        Args:
            labels: DataFrame with labeled data
            features: DataFrame with engineered features
            lookback_results: Dictionary with lookback optimization results
            regime_labels: Series with regime classifications
        
        Returns:
            ValidationReport with all check results
        """
        self.report = ValidationReport()
        
        # 1. Label autocorrelation decay
        if labels is not None:
            self._validate_label_autocorrelation(labels)
        
        # 2. Feature-target mutual information
        if features is not None and labels is not None:
            self._validate_feature_target_mi(features, labels)
        
        # 3. Feature stability across regimes
        if features is not None and regime_labels is not None:
            self._validate_feature_regime_stability(features, regime_labels)
        
        # 4. Sharpe of synthetic signal
        if features is not None and labels is not None:
            self._validate_synthetic_sharpe(features, labels)
        
        # 5. Lookback sensitivity
        if lookback_results is not None:
            self._validate_lookback_sensitivity(lookback_results)
        
        # 6. IC mean/volatility
        if features is not None and labels is not None:
            self._validate_information_coefficient(features, labels)
        
        # Generate summary
        self._generate_summary()
        
        return self.report
    
    def _validate_label_autocorrelation(
        self,
        labels: pd.DataFrame,
        max_lag: int = 10,
        threshold: float = 0.1
    ) -> None:
        """
        Test 1: Label autocorrelation decay.
        
        Ensures labels are not trivially predictable by checking that
        autocorrelation decays quickly. ρ(h) < 0.1 for h > 3
        
        Args:
            labels: DataFrame with label columns
            max_lag: Maximum lag to compute
            threshold: Maximum acceptable autocorrelation at lag > 3
        """
        try:
            label_columns = [col for col in labels.columns if 'label' in col.lower() or 'target' in col.lower()]
            
            if not label_columns:
                label_columns = labels.select_dtypes(include=[np.number]).columns[:5].tolist()
            
            if not label_columns:
                self.logger.warning("No numeric label columns found for autocorrelation test")
                self._add_result(
                    "label_autocorrelation_decay",
                    ValidationStatus.SKIPPED,
                    0.0, threshold, True,
                    "No label columns found"
                )
                return
            
            autocorr_violations = 0
            max_autocorr_above_lag3 = 0.0
            
            for col in label_columns[:3]:  # Check up to 3 label columns
                series = labels[col].dropna()
                
                if len(series) < max_lag + 10:
                    continue
                
                # Compute autocorrelations
                autocorrs = [series.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
                
                # Check lags > 3
                high_lag_autocorrs = autocorrs[3:]  # Lags 4+
                max_high_lag = max(abs(ac) for ac in high_lag_autocorrs if not np.isnan(ac))
                max_autocorr_above_lag3 = max(max_autocorr_above_lag3, max_high_lag)
                
                if max_high_lag > threshold:
                    autocorr_violations += 1
                    self.logger.warning(
                        f"Column {col} has high autocorrelation at lag>3: {max_high_lag:.4f}"
                    )
            
            passed = autocorr_violations == 0
            status = ValidationStatus.PASSED if passed else (
                ValidationStatus.WARNING if not self.strict_mode else ValidationStatus.FAILED
            )
            
            self._add_result(
                "label_autocorrelation_decay",
                status,
                max_autocorr_above_lag3,
                threshold,
                passed,
                f"Max autocorrelation (lag>3): {max_autocorr_above_lag3:.4f}, threshold: {threshold}",
                metadata={'violations': autocorr_violations, 'columns_checked': len(label_columns)}
            )
            
        except Exception as e:
            self.logger.error(f"Error in label autocorrelation test: {e}")
            self._add_result(
                "label_autocorrelation_decay",
                ValidationStatus.FAILED,
                0.0, threshold, False,
                f"Test failed with error: {e}"
            )
    
    def _validate_feature_target_mi(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        top_percentile: float = 0.10,
        min_mi_threshold: float = 0.01
    ) -> None:
        """
        Test 2: Feature-target mutual information.
        
        Filters out noise features by ensuring top features have meaningful
        mutual information with targets.
        
        Args:
            features: DataFrame with feature columns
            labels: DataFrame with label columns
            top_percentile: Top percentile of features to retain
            min_mi_threshold: Minimum MI for top features
        """
        try:
            # Use sklearn's mutual_info_regression if available
            try:
                from sklearn.feature_selection import mutual_info_regression
            except ImportError:
                self.logger.warning("sklearn not available, skipping MI test")
                self._add_result(
                    "feature_target_mutual_info",
                    ValidationStatus.SKIPPED,
                    0.0, min_mi_threshold, True,
                    "sklearn not available"
                )
                return
            
            # Get first numeric target
            target_cols = [col for col in labels.columns if 'label' in col.lower() or 'target' in col.lower()]
            if not target_cols:
                target_cols = labels.select_dtypes(include=[np.number]).columns[:1].tolist()
            
            if not target_cols:
                self._add_result(
                    "feature_target_mutual_info",
                    ValidationStatus.SKIPPED,
                    0.0, min_mi_threshold, True,
                    "No target columns found"
                )
                return
            
            target = labels[target_cols[0]].values
            
            # Select numeric features
            feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            if not feature_cols:
                self._add_result(
                    "feature_target_mutual_info",
                    ValidationStatus.SKIPPED,
                    0.0, min_mi_threshold, True,
                    "No numeric features found"
                )
                return
            
            # Align indices
            common_idx = features.index.intersection(labels.index)
            X = features.loc[common_idx, feature_cols].fillna(0).values
            y = labels.loc[common_idx, target_cols[0]].fillna(0).values
            
            if len(X) < 100:
                self._add_result(
                    "feature_target_mutual_info",
                    ValidationStatus.SKIPPED,
                    0.0, min_mi_threshold, True,
                    f"Insufficient samples: {len(X)}"
                )
                return
            
            # Compute MI
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mi_scores = mutual_info_regression(X, y, random_state=42)
            
            # Check top percentile
            n_top = max(1, int(len(mi_scores) * top_percentile))
            top_mi = np.sort(mi_scores)[-n_top:]
            mean_top_mi = np.mean(top_mi)
            
            passed = mean_top_mi >= min_mi_threshold
            status = ValidationStatus.PASSED if passed else ValidationStatus.WARNING
            
            self._add_result(
                "feature_target_mutual_info",
                status,
                mean_top_mi,
                min_mi_threshold,
                passed,
                f"Mean MI of top {top_percentile:.0%} features: {mean_top_mi:.6f}",
                metadata={
                    'n_features': len(feature_cols),
                    'n_top': n_top,
                    'max_mi': float(np.max(mi_scores)),
                    'median_mi': float(np.median(mi_scores))
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in feature-target MI test: {e}")
            self._add_result(
                "feature_target_mutual_info",
                ValidationStatus.FAILED,
                0.0, min_mi_threshold, False,
                f"Test failed with error: {e}"
            )
    
    def _validate_feature_regime_stability(
        self,
        features: pd.DataFrame,
        regime_labels: pd.Series,
        ks_threshold: float = 0.05
    ) -> None:
        """
        Test 3: Feature stability across regimes.
        
        Ensures features maintain similar distributions across regimes
        using Kolmogorov-Smirnov test.
        
        Args:
            features: DataFrame with feature columns
            regime_labels: Series with regime classifications
            ks_threshold: KS test p-value threshold (p > 0.05 means stable)
        """
        try:
            regimes = regime_labels.unique()
            
            if len(regimes) < 2:
                self._add_result(
                    "feature_regime_stability",
                    ValidationStatus.SKIPPED,
                    0.0, ks_threshold, True,
                    "Less than 2 regimes found"
                )
                return
            
            feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()[:20]  # Sample 20 features
            
            if not feature_cols:
                self._add_result(
                    "feature_regime_stability",
                    ValidationStatus.SKIPPED,
                    0.0, ks_threshold, True,
                    "No numeric features found"
                )
                return
            
            # Align indices
            common_idx = features.index.intersection(regime_labels.index)
            features_aligned = features.loc[common_idx]
            regimes_aligned = regime_labels.loc[common_idx]
            
            unstable_features = 0
            min_p_value = 1.0
            
            for col in feature_cols:
                feature_data = features_aligned[col].dropna()
                
                # Compare first two regimes
                regime1_data = feature_data[regimes_aligned == regimes[0]]
                regime2_data = feature_data[regimes_aligned == regimes[1]]
                
                if len(regime1_data) < 30 or len(regime2_data) < 30:
                    continue
                
                # KS test
                ks_stat, p_value = ks_2samp(regime1_data, regime2_data)
                min_p_value = min(min_p_value, p_value)
                
                if p_value < ks_threshold:
                    unstable_features += 1
            
            # Pass if < 30% features are unstable
            instability_rate = unstable_features / len(feature_cols) if feature_cols else 0
            passed = instability_rate < 0.30
            
            status = ValidationStatus.PASSED if passed else ValidationStatus.WARNING
            
            self._add_result(
                "feature_regime_stability",
                status,
                min_p_value,
                ks_threshold,
                passed,
                f"Feature instability rate: {instability_rate:.2%}, min p-value: {min_p_value:.4f}",
                metadata={
                    'unstable_features': unstable_features,
                    'total_features': len(feature_cols),
                    'n_regimes': len(regimes)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in feature regime stability test: {e}")
            self._add_result(
                "feature_regime_stability",
                ValidationStatus.FAILED,
                0.0, ks_threshold, False,
                f"Test failed with error: {e}"
            )
    
    def _validate_synthetic_sharpe(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        min_sharpe: float = 0.5
    ) -> None:
        """
        Test 4: Sharpe of synthetic signal.
        
        Creates a simple linear signal from top features and validates
        it has economically plausible Sharpe ratio.
        
        Args:
            features: DataFrame with feature columns
            labels: DataFrame with label columns
            min_sharpe: Minimum acceptable Sharpe ratio
        """
        try:
            # Get target
            target_cols = [col for col in labels.columns if 'label' in col.lower() or 'target' in col.lower()]
            if not target_cols:
                target_cols = labels.select_dtypes(include=[np.number]).columns[:1].tolist()
            
            if not target_cols:
                self._add_result(
                    "synthetic_signal_sharpe",
                    ValidationStatus.SKIPPED,
                    0.0, min_sharpe, True,
                    "No target columns found"
                )
                return
            
            # Select top correlated features
            feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(feature_cols) < 3:
                self._add_result(
                    "synthetic_signal_sharpe",
                    ValidationStatus.SKIPPED,
                    0.0, min_sharpe, True,
                    f"Insufficient features: {len(feature_cols)}"
                )
                return
            
            # Align indices
            common_idx = features.index.intersection(labels.index)
            X = features.loc[common_idx, feature_cols].fillna(0)
            y = labels.loc[common_idx, target_cols[0]].fillna(0)
            
            if len(X) < 100:
                self._add_result(
                    "synthetic_signal_sharpe",
                    ValidationStatus.SKIPPED,
                    0.0, min_sharpe, True,
                    f"Insufficient samples: {len(X)}"
                )
                return
            
            # Select top 5 features by correlation
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            top_features = correlations.head(5).index.tolist()
            
            # Create simple weighted signal
            weights = correlations[top_features].values
            weights = weights / weights.sum()
            
            signal = (X[top_features] * weights).sum(axis=1)
            
            # Compute returns (signal * target)
            returns = signal * y
            returns = returns.replace([np.inf, -np.inf], 0).fillna(0)
            
            # Compute Sharpe ratio
            if returns.std() > 1e-8:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
            else:
                sharpe = 0.0
            
            passed = sharpe >= min_sharpe
            status = ValidationStatus.PASSED if passed else ValidationStatus.WARNING
            
            self._add_result(
                "synthetic_signal_sharpe",
                status,
                sharpe,
                min_sharpe,
                passed,
                f"Synthetic signal Sharpe ratio: {sharpe:.4f}",
                metadata={
                    'n_features_used': len(top_features),
                    'mean_return': float(returns.mean()),
                    'volatility': float(returns.std())
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in synthetic Sharpe test: {e}")
            self._add_result(
                "synthetic_signal_sharpe",
                ValidationStatus.FAILED,
                0.0, min_sharpe, False,
                f"Test failed with error: {e}"
            )
    
    def _validate_lookback_sensitivity(
        self,
        lookback_results: Dict[str, Any],
        max_change_threshold: float = 0.15
    ) -> None:
        """
        Test 5: Lookback sensitivity.
        
        Ensures optimal lookback windows are robust to resampling
        (< 15% change under bootstrap).
        
        Args:
            lookback_results: Dictionary with lookback optimization results
            max_change_threshold: Maximum acceptable change in lookback
        """
        try:
            if 'optimal_lookback' not in lookback_results:
                self._add_result(
                    "lookback_sensitivity",
                    ValidationStatus.SKIPPED,
                    0.0, max_change_threshold, True,
                    "No optimal_lookback in results"
                )
                return
            
            optimal = lookback_results.get('optimal_lookback', 0)
            
            # Check if we have resampling results
            resampled = lookback_results.get('resampled_lookbacks', [])
            
            if not resampled or len(resampled) < 3:
                # No resampling data, check stability from metadata
                stability = lookback_results.get('stability_score', 0.0)
                passed = stability > 0.7  # High stability
                
                self._add_result(
                    "lookback_sensitivity",
                    ValidationStatus.WARNING if not passed else ValidationStatus.PASSED,
                    stability,
                    0.7,
                    passed,
                    f"Lookback stability score: {stability:.4f} (no resampling data)",
                    metadata={'optimal_lookback': optimal}
                )
                return
            
            # Compute variability of resampled lookbacks
            resampled_array = np.array(resampled)
            mean_resampled = np.mean(resampled_array)
            std_resampled = np.std(resampled_array)
            
            # Relative change
            relative_change = std_resampled / (mean_resampled + 1e-8)
            
            passed = relative_change < max_change_threshold
            status = ValidationStatus.PASSED if passed else ValidationStatus.WARNING
            
            self._add_result(
                "lookback_sensitivity",
                status,
                relative_change,
                max_change_threshold,
                passed,
                f"Lookback sensitivity: {relative_change:.4f}, optimal: {optimal}",
                metadata={
                    'optimal_lookback': optimal,
                    'mean_resampled': float(mean_resampled),
                    'std_resampled': float(std_resampled),
                    'n_resamples': len(resampled)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in lookback sensitivity test: {e}")
            self._add_result(
                "lookback_sensitivity",
                ValidationStatus.FAILED,
                0.0, max_change_threshold, False,
                f"Test failed with error: {e}"
            )
    
    def _validate_information_coefficient(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        min_mean_ic: float = 0.02,
        min_t_stat: float = 2.0
    ) -> None:
        """
        Test 6: IC (Information Coefficient) mean and volatility.
        
        Validates predictive quality of features using rank correlation
        with forward returns. Mean(IC) ≈ 0.02-0.05, t-stat > 2
        
        Args:
            features: DataFrame with feature columns
            labels: DataFrame with label columns
            min_mean_ic: Minimum mean IC
            min_t_stat: Minimum t-statistic
        """
        try:
            # Get target
            target_cols = [col for col in labels.columns if 'label' in col.lower() or 'target' in col.lower()]
            if not target_cols:
                target_cols = labels.select_dtypes(include=[np.number]).columns[:1].tolist()
            
            if not target_cols:
                self._add_result(
                    "information_coefficient",
                    ValidationStatus.SKIPPED,
                    0.0, min_mean_ic, True,
                    "No target columns found"
                )
                return
            
            feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()[:30]  # Sample 30 features
            
            if not feature_cols:
                self._add_result(
                    "information_coefficient",
                    ValidationStatus.SKIPPED,
                    0.0, min_mean_ic, True,
                    "No numeric features found"
                )
                return
            
            # Align indices
            common_idx = features.index.intersection(labels.index)
            X = features.loc[common_idx, feature_cols]
            y = labels.loc[common_idx, target_cols[0]]
            
            if len(X) < 100:
                self._add_result(
                    "information_coefficient",
                    ValidationStatus.SKIPPED,
                    0.0, min_mean_ic, True,
                    f"Insufficient samples: {len(X)}"
                )
                return
            
            # Compute IC for each feature (rank correlation)
            ics = []
            for col in feature_cols:
                feature_data = X[col].dropna()
                target_data = y.loc[feature_data.index].dropna()
                
                if len(feature_data) < 50:
                    continue
                
                # Rank correlation (Spearman)
                try:
                    ic, _ = stats.spearmanr(feature_data, target_data, nan_policy='omit')
                    if not np.isnan(ic):
                        ics.append(ic)
                except Exception:
                    continue
            
            if not ics:
                self._add_result(
                    "information_coefficient",
                    ValidationStatus.SKIPPED,
                    0.0, min_mean_ic, True,
                    "Could not compute ICs"
                )
                return
            
            # Compute statistics
            mean_ic = np.mean(ics)
            std_ic = np.std(ics)
            t_stat = mean_ic / (std_ic / np.sqrt(len(ics)) + 1e-8)
            
            passed = mean_ic >= min_mean_ic and t_stat >= min_t_stat
            status = ValidationStatus.PASSED if passed else ValidationStatus.WARNING
            
            self._add_result(
                "information_coefficient",
                status,
                mean_ic,
                min_mean_ic,
                passed,
                f"Mean IC: {mean_ic:.4f}, t-stat: {t_stat:.2f}",
                metadata={
                    'mean_ic': float(mean_ic),
                    'std_ic': float(std_ic),
                    't_stat': float(t_stat),
                    'n_features': len(ics)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in IC test: {e}")
            self._add_result(
                "information_coefficient",
                ValidationStatus.FAILED,
                0.0, min_mean_ic, False,
                f"Test failed with error: {e}"
            )
    
    def _add_result(
        self,
        test_name: str,
        status: ValidationStatus,
        value: float,
        threshold: float,
        passed: bool,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a validation result to the report."""
        result = ValidationResult(
            test_name=test_name,
            status=status,
            value=value,
            threshold=threshold,
            passed=passed,
            message=message,
            metadata=metadata or {}
        )
        self.report.results.append(result)
        
        # Log result
        log_level = logging.INFO if passed else (
            logging.WARNING if status == ValidationStatus.WARNING else logging.ERROR
        )
        self.logger.log(log_level, f"[{test_name}] {message}")
    
    def _generate_summary(self) -> None:
        """Generate summary statistics for the report."""
        self.report.summary = {
            'total_tests': len(self.report.results),
            'passed': sum(1 for r in self.report.results if r.status == ValidationStatus.PASSED),
            'warnings': self.report.warnings_count,
            'failures': self.report.failures_count,
            'skipped': sum(1 for r in self.report.results if r.status == ValidationStatus.SKIPPED),
            'overall_passed': self.report.passed
        }
        
        self.logger.info(
            f"Validation complete: {self.report.summary['passed']}/{self.report.summary['total_tests']} passed, "
            f"{self.report.warnings_count} warnings, {self.report.failures_count} failures"
        )


def validate_pre_training_outputs(
    labels: Optional[pd.DataFrame] = None,
    features: Optional[pd.DataFrame] = None,
    lookback_results: Optional[Dict[str, Any]] = None,
    regime_labels: Optional[pd.Series] = None,
    strict_mode: bool = False,
    logger: Optional[logging.Logger] = None
) -> ValidationReport:
    """
    Convenience function to validate pre-training pipeline outputs.
    
    Args:
        labels: DataFrame with labeled data
        features: DataFrame with engineered features
        lookback_results: Dictionary with lookback optimization results
        regime_labels: Series with regime classifications
        strict_mode: If True, warnings are treated as failures
        logger: Optional logger instance
    
    Returns:
        ValidationReport with all check results
    """
    validator = QuantitativeValidator(logger=logger, strict_mode=strict_mode)
    return validator.validate_all(
        labels=labels,
        features=features,
        lookback_results=lookback_results,
        regime_labels=regime_labels
    )


__all__ = [
    'QuantitativeValidator',
    'ValidationResult',
    'ValidationReport',
    'ValidationStatus',
    'validate_pre_training_outputs',
]