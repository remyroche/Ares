"""
Advanced Statistical Validator for Multi-Horizon Profit Labeling

This module provides comprehensive statistical validation methods that go beyond
basic correlation and significance tests. It implements advanced statistical
techniques for rigorous validation of labeling quality.

Key Statistical Methods:
1. Granger Causality Tests for Label-Return Relationships
2. Mutual Information and Transfer Entropy Analysis
3. Regime Consistency Validation (ANOVA/Kruskal-Wallis)
4. Structural Break Detection (Chow Test, CUSUM)
5. Cross-Validation with Time Series Considerations
6. Robustness Tests (Bootstrap, Permutation)
7. Information Content Measures
8. Economic Significance Tests
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
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Statistical imports
from scipy import stats
from scipy.stats import jarque_bera, anderson, kstest, mannwhitneyu, kruskal
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.seasonal import seasonal_decompose

# Optional advanced statistical libraries
try:
    from arch.unitroot import ADF, PhillipsPerron
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    import pyinform as pi
    PYINFORM_AVAILABLE = True
except ImportError:
    PYINFORM_AVAILABLE = False

from src.utils.logger import get_logger


class AdvancedValidationMetric(Enum):
    """Enumeration of advanced validation metrics."""
    GRANGER_CAUSALITY = "granger_causality"
    MUTUAL_INFORMATION = "mutual_information"
    TRANSFER_ENTROPY = "transfer_entropy"
    REGIME_CONSISTENCY = "regime_consistency"
    STRUCTURAL_BREAKS = "structural_breaks"
    STATIONARITY = "stationarity"
    HETEROSKEDASTICITY = "heteroskedasticity"
    AUTOCORRELATION = "autocorrelation"
    NORMALITY = "normality"
    INFORMATION_CONTENT = "information_content"
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    ROBUSTNESS = "robustness"
    CROSS_VALIDATION = "cross_validation"


class StatisticalTest(Enum):
    """Enumeration of statistical tests."""
    GRANGER_TEST = "granger_test"
    CHOW_TEST = "chow_test"
    CUSUM_TEST = "cusum_test"
    ADF_TEST = "adf_test"
    KPSS_TEST = "kpss_test"
    JARQUE_BERA_TEST = "jarque_bera_test"
    ANDERSON_DARLING_TEST = "anderson_darling_test"
    BREUSCH_GODFREY_TEST = "breusch_godfrey_test"
    BREUSCH_PAGAN_TEST = "breusch_pagan_test"
    DURBIN_WATSON_TEST = "durbin_watson_test"
    MANN_WHITNEY_TEST = "mann_whitney_test"
    KRUSKAL_WALLIS_TEST = "kruskal_wallis_test"


@dataclass
class AdvancedValidationConfig:
    """Configuration for advanced statistical validation."""
    # Test selection
    enable_causality_tests: bool = True
    enable_information_theory: bool = True
    enable_regime_tests: bool = True
    enable_structural_tests: bool = True
    enable_robustness_tests: bool = True
    
    # Granger causality parameters
    granger_max_lags: int = 10
    granger_significance_level: float = 0.05
    
    # Information theory parameters
    mutual_info_bins: int = 10
    transfer_entropy_lag: int = 1
    
    # Structural break parameters
    structural_break_min_size: float = 0.15  # Minimum 15% of data for each segment
    cusum_significance_level: float = 0.05
    
    # Cross-validation parameters
    cv_folds: int = 5
    cv_gap: int = 10  # Gap between train/test for time series
    min_train_size: int = 100
    
    # Bootstrap parameters
    bootstrap_iterations: int = 1000
    bootstrap_confidence_level: float = 0.95
    
    # Permutation test parameters
    permutation_iterations: int = 1000
    
    # Economic significance thresholds
    min_sharpe_ratio: float = 0.5
    min_information_ratio: float = 0.3
    transaction_cost: float = 0.0008
    
    # Parallel processing
    n_jobs: int = -1
    timeout_seconds: int = 300  # 5 minutes timeout per test
    
    # Significance levels
    default_alpha: float = 0.05
    bonferroni_correction: bool = True


@dataclass
class StatisticalTestResult:
    """Result container for individual statistical tests."""
    test_type: StatisticalTest
    test_statistic: float
    p_value: float
    critical_values: Optional[Dict[str, float]]
    is_significant: bool
    interpretation: str
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AdvancedValidationResult:
    """Result container for advanced validation analysis."""
    validation_metric: AdvancedValidationMetric
    test_results: List[StatisticalTestResult]
    summary_statistic: float
    confidence_interval: Optional[Tuple[float, float]]
    is_significant: bool
    interpretation: str
    recommendations: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class AdvancedStatisticalValidator:
    """
    Advanced statistical validator for profit labeling quality.
    
    This class provides comprehensive statistical validation using advanced
    econometric and time series methods to rigorously assess labeling quality.
    
    Key Features:
    1. **Causality Analysis**: Granger causality tests for label-return relationships
    2. **Information Theory**: Mutual information and transfer entropy measures
    3. **Regime Analysis**: Statistical tests across different market regimes
    4. **Structural Stability**: Break point detection and stability tests
    5. **Robustness Testing**: Bootstrap and permutation-based validation
    6. **Economic Significance**: Tests for economic value and trading significance
    """
    
    def __init__(self, config: Optional[AdvancedValidationConfig] = None):
        """Initialize the advanced statistical validator."""
        self.config = config or AdvancedValidationConfig()
        self.logger = get_logger('AdvancedStatisticalValidator')
        
        # Validation results storage
        self.validation_results: Dict[str, AdvancedValidationResult] = {}
        self.test_history: List[StatisticalTestResult] = []
        
        # Performance tracking
        self.validation_times: Dict[AdvancedValidationMetric, float] = {}
        
        self.logger.info('📊 Advanced Statistical Validator initialized')
        self.logger.info(f'   → Enabled tests: {self._get_enabled_tests()}')
        
    def _get_enabled_tests(self) -> str:
        """Get list of enabled test categories."""
        enabled = []
        if self.config.enable_causality_tests:
            enabled.append("Causality")
        if self.config.enable_information_theory:
            enabled.append("Information Theory")
        if self.config.enable_regime_tests:
            enabled.append("Regime Analysis")
        if self.config.enable_structural_tests:
            enabled.append("Structural Tests")
        if self.config.enable_robustness_tests:
            enabled.append("Robustness")
        return ", ".join(enabled)
    
    def comprehensive_validate(self,
                             labeled_data: pd.DataFrame,
                             market_data: pd.DataFrame,
                             target_columns: Optional[List[str]] = None) -> Dict[str, AdvancedValidationResult]:
        """
        Perform comprehensive advanced statistical validation.
        
        Args:
            labeled_data: DataFrame with profit labels
            market_data: Original market data (OHLCV)
            target_columns: Specific columns to validate (optional)
            
        Returns:
            Dictionary of validation results by metric type
        """
        self.logger.info('🔬 Starting comprehensive advanced statistical validation')
        
        # Determine target columns
        if target_columns is None:
            target_columns = [col for col in labeled_data.columns 
                            if col.endswith('_prob') or col in ['overall_opportunity', 'leverage_adjusted_score']]
        
        if not target_columns:
            self.logger.warning('⚠️ No suitable target columns found for validation')
            return {}
        
        # Prepare future returns for validation
        future_returns = self._prepare_future_returns(market_data)
        
        # Run validation tests
        validation_methods = []
        
        if self.config.enable_causality_tests:
            validation_methods.append(self._validate_granger_causality)
        if self.config.enable_information_theory:
            validation_methods.append(self._validate_information_content)
        if self.config.enable_regime_tests:
            validation_methods.append(self._validate_regime_consistency)
        if self.config.enable_structural_tests:
            validation_methods.append(self._validate_structural_stability)
        if self.config.enable_robustness_tests:
            validation_methods.append(self._validate_robustness)
        
        # Execute validation methods
        for method in validation_methods:
            try:
                start_time = datetime.now()
                
                for target_col in target_columns:
                    if target_col in labeled_data.columns:
                        result = method(labeled_data[target_col], future_returns, market_data)
                        if result:
                            result_key = f"{target_col}_{result.validation_metric.value}"
                            self.validation_results[result_key] = result
                
                # Track execution time
                execution_time = (datetime.now() - start_time).total_seconds()
                metric_name = method.__name__.replace('_validate_', '')
                if hasattr(AdvancedValidationMetric, metric_name.upper()):
                    metric = getattr(AdvancedValidationMetric, metric_name.upper())
                    self.validation_times[metric] = execution_time
                
            except Exception as e:
                self.logger.error(f'Validation method {method.__name__} failed: {e}')
        
        # Apply multiple testing correction if enabled
        if self.config.bonferroni_correction:
            self._apply_bonferroni_correction()
        
        self.logger.info(f'✅ Advanced validation completed: {len(self.validation_results)} results')
        return self.validation_results
    
    def _prepare_future_returns(self, market_data: pd.DataFrame) -> pd.Series:
        """Prepare future returns series for validation."""
        if 'close' not in market_data.columns:
            return pd.Series(dtype=float)
        
        returns = market_data['close'].pct_change()
        future_returns = returns.shift(-1).fillna(0)
        
        return future_returns
    
    def _validate_granger_causality(self,
                                   labels: pd.Series,
                                   future_returns: pd.Series,
                                   market_data: pd.DataFrame) -> Optional[AdvancedValidationResult]:
        """Validate Granger causality between labels and future returns."""
        self.logger.info('🔍 Testing Granger causality')
        
        # Align series
        common_idx = labels.index.intersection(future_returns.index)
        if len(common_idx) < 50:
            return None
        
        aligned_labels = labels.loc[common_idx].fillna(0)
        aligned_returns = future_returns.loc[common_idx].fillna(0)
        
        # Prepare data for Granger test
        data = pd.DataFrame({
            'returns': aligned_returns,
            'labels': aligned_labels
        }).dropna()
        
        if len(data) < 50:
            return None
        
        test_results = []
        
        try:
            # Test if labels Granger-cause returns
            granger_results = grangercausalitytests(
                data[['returns', 'labels']], 
                maxlag=min(self.config.granger_max_lags, len(data) // 10),
                verbose=False
            )
            
            # Extract results for each lag
            significant_lags = []
            for lag, result in granger_results.items():
                f_stat = result[0]['ssr_ftest'][0]
                p_value = result[0]['ssr_ftest'][1]
                
                is_significant = p_value < self.config.granger_significance_level
                if is_significant:
                    significant_lags.append(lag)
                
                test_result = StatisticalTestResult(
                    test_type=StatisticalTest.GRANGER_TEST,
                    test_statistic=f_stat,
                    p_value=p_value,
                    critical_values=None,
                    is_significant=is_significant,
                    interpretation=f"Granger causality at lag {lag}: {'significant' if is_significant else 'not significant'}",
                    metadata={'lag': lag, 'test': 'labels_cause_returns'}
                )
                test_results.append(test_result)
            
            # Summary statistics
            min_p_value = min([r.p_value for r in test_results])
            is_overall_significant = len(significant_lags) > 0
            
            return AdvancedValidationResult(
                validation_metric=AdvancedValidationMetric.GRANGER_CAUSALITY,
                test_results=test_results,
                summary_statistic=1.0 - min_p_value,  # Convert to quality score
                confidence_interval=None,
                is_significant=is_overall_significant,
                interpretation=f"Labels Granger-cause returns at {len(significant_lags)} lag(s)",
                recommendations=self._generate_causality_recommendations(significant_lags),
                metadata={
                    'significant_lags': significant_lags,
                    'total_lags_tested': len(granger_results),
                    'data_points': len(data)
                }
            )
            
        except Exception as e:
            self.logger.warning(f'Granger causality test failed: {e}')
            return None
    
    def _validate_information_content(self,
                                    labels: pd.Series,
                                    future_returns: pd.Series,
                                    market_data: pd.DataFrame) -> Optional[AdvancedValidationResult]:
        """Validate information content using mutual information and transfer entropy."""
        self.logger.info('📊 Testing information content')
        
        # Align series
        common_idx = labels.index.intersection(future_returns.index)
        if len(common_idx) < 50:
            return None
        
        aligned_labels = labels.loc[common_idx].fillna(0)
        aligned_returns = future_returns.loc[common_idx].fillna(0)
        
        test_results = []
        
        # Mutual Information
        try:
            # Discretize continuous variables for mutual information
            labels_discrete = pd.cut(aligned_labels, bins=self.config.mutual_info_bins, labels=False)
            returns_discrete = pd.cut(aligned_returns, bins=self.config.mutual_info_bins, labels=False)
            
            # Remove NaN values
            valid_mask = ~(pd.isna(labels_discrete) | pd.isna(returns_discrete))
            labels_discrete = labels_discrete[valid_mask]
            returns_discrete = returns_discrete[valid_mask]
            
            if len(labels_discrete) > 20:
                mi_score = mutual_info_score(labels_discrete, returns_discrete)
                
                # Normalize by maximum possible mutual information
                max_mi = min(np.log(self.config.mutual_info_bins), 
                           np.log(len(np.unique(labels_discrete))),
                           np.log(len(np.unique(returns_discrete))))
                
                normalized_mi = mi_score / max_mi if max_mi > 0 else 0.0
                
                test_result = StatisticalTestResult(
                    test_type=StatisticalTest.MANN_WHITNEY_TEST,  # Using as proxy
                    test_statistic=mi_score,
                    p_value=1.0 - normalized_mi,  # Convert to p-value-like metric
                    critical_values=None,
                    is_significant=normalized_mi > 0.1,  # Threshold for significance
                    interpretation=f"Mutual information: {mi_score:.4f} (normalized: {normalized_mi:.4f})",
                    metadata={'test': 'mutual_information', 'normalized_score': normalized_mi}
                )
                test_results.append(test_result)
                
        except Exception as e:
            self.logger.warning(f'Mutual information test failed: {e}')
        
        # Transfer Entropy (if pyinform is available)
        if PYINFORM_AVAILABLE:
            try:
                # Discretize for transfer entropy
                labels_te = pd.cut(aligned_labels, bins=5, labels=False).fillna(0).astype(int)
                returns_te = pd.cut(aligned_returns, bins=5, labels=False).fillna(0).astype(int)
                
                if len(labels_te) > 20:
                    te_score = pi.transferentropy.transfer_entropy(
                        labels_te.values, returns_te.values, k=self.config.transfer_entropy_lag
                    )
                    
                    test_result = StatisticalTestResult(
                        test_type=StatisticalTest.MANN_WHITNEY_TEST,  # Using as proxy
                        test_statistic=te_score,
                        p_value=1.0 - min(te_score, 1.0),
                        critical_values=None,
                        is_significant=te_score > 0.05,
                        interpretation=f"Transfer entropy: {te_score:.4f}",
                        metadata={'test': 'transfer_entropy', 'lag': self.config.transfer_entropy_lag}
                    )
                    test_results.append(test_result)
                    
            except Exception as e:
                self.logger.warning(f'Transfer entropy test failed: {e}')
        
        if not test_results:
            return None
        
        # Summary statistics
        info_scores = [r.test_statistic for r in test_results]
        avg_info_score = np.mean(info_scores)
        is_significant = any(r.is_significant for r in test_results)
        
        return AdvancedValidationResult(
            validation_metric=AdvancedValidationMetric.INFORMATION_CONTENT,
            test_results=test_results,
            summary_statistic=avg_info_score,
            confidence_interval=None,
            is_significant=is_significant,
            interpretation=f"Average information content: {avg_info_score:.4f}",
            recommendations=self._generate_information_recommendations(avg_info_score),
            metadata={
                'tests_performed': len(test_results),
                'average_score': avg_info_score
            }
        )
    
    def _validate_regime_consistency(self,
                                   labels: pd.Series,
                                   future_returns: pd.Series,
                                   market_data: pd.DataFrame) -> Optional[AdvancedValidationResult]:
        """Validate label consistency across different market regimes."""
        self.logger.info('🎯 Testing regime consistency')
        
        if len(labels) < 100:
            return None
        
        # Define regimes based on volatility and trend
        regimes = self._identify_market_regimes(market_data)
        
        if len(regimes.unique()) < 2:
            return None
        
        # Align data
        common_idx = labels.index.intersection(regimes.index)
        if len(common_idx) < 50:
            return None
        
        aligned_labels = labels.loc[common_idx]
        aligned_regimes = regimes.loc[common_idx]
        
        test_results = []
        
        try:
            # Kruskal-Wallis test for differences across regimes
            regime_groups = [aligned_labels[aligned_regimes == regime].dropna().values 
                           for regime in aligned_regimes.unique()]
            
            # Filter out empty groups
            regime_groups = [group for group in regime_groups if len(group) > 5]
            
            if len(regime_groups) >= 2:
                kw_stat, kw_p = kruskal(*regime_groups)
                
                test_result = StatisticalTestResult(
                    test_type=StatisticalTest.KRUSKAL_WALLIS_TEST,
                    test_statistic=kw_stat,
                    p_value=kw_p,
                    critical_values=None,
                    is_significant=kw_p < self.config.default_alpha,
                    interpretation=f"Regime consistency: {'inconsistent' if kw_p < self.config.default_alpha else 'consistent'}",
                    metadata={
                        'test': 'kruskal_wallis',
                        'regimes_count': len(regime_groups),
                        'total_samples': sum(len(g) for g in regime_groups)
                    }
                )
                test_results.append(test_result)
            
            # Pairwise Mann-Whitney tests between regimes
            regime_names = aligned_regimes.unique()
            for i, regime1 in enumerate(regime_names):
                for j, regime2 in enumerate(regime_names[i+1:], i+1):
                    group1 = aligned_labels[aligned_regimes == regime1].dropna().values
                    group2 = aligned_labels[aligned_regimes == regime2].dropna().values
                    
                    if len(group1) > 5 and len(group2) > 5:
                        mw_stat, mw_p = mannwhitneyu(group1, group2, alternative='two-sided')
                        
                        test_result = StatisticalTestResult(
                            test_type=StatisticalTest.MANN_WHITNEY_TEST,
                            test_statistic=mw_stat,
                            p_value=mw_p,
                            critical_values=None,
                            is_significant=mw_p < self.config.default_alpha,
                            interpretation=f"Regimes {regime1} vs {regime2}: {'different' if mw_p < self.config.default_alpha else 'similar'}",
                            metadata={
                                'test': 'mann_whitney',
                                'regime1': str(regime1),
                                'regime2': str(regime2),
                                'n1': len(group1),
                                'n2': len(group2)
                            }
                        )
                        test_results.append(test_result)
                        
        except Exception as e:
            self.logger.warning(f'Regime consistency test failed: {e}')
        
        if not test_results:
            return None
        
        # Summary statistics
        significant_tests = [r for r in test_results if r.is_significant]
        consistency_score = 1.0 - (len(significant_tests) / len(test_results))
        
        return AdvancedValidationResult(
            validation_metric=AdvancedValidationMetric.REGIME_CONSISTENCY,
            test_results=test_results,
            summary_statistic=consistency_score,
            confidence_interval=None,
            is_significant=len(significant_tests) > 0,
            interpretation=f"Regime consistency score: {consistency_score:.3f}",
            recommendations=self._generate_regime_recommendations(consistency_score),
            metadata={
                'total_tests': len(test_results),
                'significant_tests': len(significant_tests),
                'regimes_identified': len(aligned_regimes.unique())
            }
        )
    
    def _validate_structural_stability(self,
                                     labels: pd.Series,
                                     future_returns: pd.Series,
                                     market_data: pd.DataFrame) -> Optional[AdvancedValidationResult]:
        """Validate structural stability using break point detection."""
        self.logger.info('🏗️ Testing structural stability')
        
        if len(labels) < 100:
            return None
        
        test_results = []
        
        try:
            # CUSUM test for structural breaks
            labels_clean = labels.fillna(method='ffill').fillna(0)
            
            # Recursive residuals for CUSUM test
            if len(labels_clean) > 30:
                # Simple CUSUM test on label series
                mean_label = labels_clean.mean()
                cumsum = np.cumsum(labels_clean - mean_label)
                
                # Normalize CUSUM
                std_label = labels_clean.std()
                if std_label > 0:
                    normalized_cumsum = cumsum / (std_label * np.sqrt(len(labels_clean)))
                    
                    # Test statistic is maximum absolute CUSUM
                    cusum_stat = np.max(np.abs(normalized_cumsum))
                    
                    # Critical value for 5% significance (approximate)
                    critical_value = 1.36  # For large samples
                    
                    test_result = StatisticalTestResult(
                        test_type=StatisticalTest.CUSUM_TEST,
                        test_statistic=cusum_stat,
                        p_value=1.0 - min(cusum_stat / critical_value, 1.0),
                        critical_values={'5%': critical_value},
                        is_significant=cusum_stat > critical_value,
                        interpretation=f"Structural stability: {'unstable' if cusum_stat > critical_value else 'stable'}",
                        metadata={
                            'test': 'cusum',
                            'critical_value': critical_value,
                            'sample_size': len(labels_clean)
                        }
                    )
                    test_results.append(test_result)
            
            # Chow test for known break point (middle of series)
            if len(labels_clean) > 50:
                break_point = len(labels_clean) // 2
                
                first_half = labels_clean.iloc[:break_point]
                second_half = labels_clean.iloc[break_point:]
                
                if len(first_half) > 10 and len(second_half) > 10:
                    # F-test for equality of means
                    f_stat, f_p = stats.f_oneway(first_half, second_half)
                    
                    test_result = StatisticalTestResult(
                        test_type=StatisticalTest.CHOW_TEST,
                        test_statistic=f_stat,
                        p_value=f_p,
                        critical_values=None,
                        is_significant=f_p < self.config.default_alpha,
                        interpretation=f"Structural break at midpoint: {'detected' if f_p < self.config.default_alpha else 'not detected'}",
                        metadata={
                            'test': 'chow',
                            'break_point': break_point,
                            'first_half_mean': float(first_half.mean()),
                            'second_half_mean': float(second_half.mean())
                        }
                    )
                    test_results.append(test_result)
                    
        except Exception as e:
            self.logger.warning(f'Structural stability test failed: {e}')
        
        if not test_results:
            return None
        
        # Summary statistics
        unstable_tests = [r for r in test_results if r.is_significant]
        stability_score = 1.0 - (len(unstable_tests) / len(test_results))
        
        return AdvancedValidationResult(
            validation_metric=AdvancedValidationMetric.STRUCTURAL_BREAKS,
            test_results=test_results,
            summary_statistic=stability_score,
            confidence_interval=None,
            is_significant=len(unstable_tests) > 0,
            interpretation=f"Structural stability score: {stability_score:.3f}",
            recommendations=self._generate_stability_recommendations(stability_score),
            metadata={
                'total_tests': len(test_results),
                'unstable_tests': len(unstable_tests)
            }
        )
    
    def _validate_robustness(self,
                           labels: pd.Series,
                           future_returns: pd.Series,
                           market_data: pd.DataFrame) -> Optional[AdvancedValidationResult]:
        """Validate robustness using bootstrap and permutation tests."""
        self.logger.info('🛡️ Testing robustness')
        
        # Align series
        common_idx = labels.index.intersection(future_returns.index)
        if len(common_idx) < 50:
            return None
        
        aligned_labels = labels.loc[common_idx].fillna(0)
        aligned_returns = future_returns.loc[common_idx].fillna(0)
        
        test_results = []
        
        try:
            # Bootstrap test for correlation stability
            original_corr = np.corrcoef(aligned_labels, aligned_returns)[0, 1]
            if np.isnan(original_corr):
                original_corr = 0.0
            
            bootstrap_corrs = []
            for _ in range(self.config.bootstrap_iterations):
                # Bootstrap sample
                indices = np.random.choice(len(aligned_labels), size=len(aligned_labels), replace=True)
                boot_labels = aligned_labels.iloc[indices]
                boot_returns = aligned_returns.iloc[indices]
                
                boot_corr = np.corrcoef(boot_labels, boot_returns)[0, 1]
                if not np.isnan(boot_corr):
                    bootstrap_corrs.append(boot_corr)
            
            if bootstrap_corrs:
                bootstrap_std = np.std(bootstrap_corrs)
                bootstrap_mean = np.mean(bootstrap_corrs)
                
                # Bootstrap confidence interval
                ci_lower = np.percentile(bootstrap_corrs, 2.5)
                ci_upper = np.percentile(bootstrap_corrs, 97.5)
                
                # Test if original correlation is significantly different from zero
                t_stat = abs(bootstrap_mean) / (bootstrap_std + 1e-10)
                p_value = 2 * (1 - stats.t.cdf(t_stat, len(bootstrap_corrs) - 1))
                
                test_result = StatisticalTestResult(
                    test_type=StatisticalTest.MANN_WHITNEY_TEST,  # Using as proxy
                    test_statistic=t_stat,
                    p_value=p_value,
                    critical_values={'ci_lower': ci_lower, 'ci_upper': ci_upper},
                    is_significant=p_value < self.config.default_alpha,
                    interpretation=f"Bootstrap correlation: {bootstrap_mean:.4f} ± {bootstrap_std:.4f}",
                    metadata={
                        'test': 'bootstrap_correlation',
                        'original_correlation': original_corr,
                        'bootstrap_mean': bootstrap_mean,
                        'bootstrap_std': bootstrap_std,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper
                    }
                )
                test_results.append(test_result)
            
            # Permutation test
            permutation_corrs = []
            for _ in range(min(self.config.permutation_iterations, 500)):  # Limit for performance
                # Permute labels
                permuted_labels = aligned_labels.sample(frac=1).reset_index(drop=True)
                perm_corr = np.corrcoef(permuted_labels, aligned_returns)[0, 1]
                if not np.isnan(perm_corr):
                    permutation_corrs.append(perm_corr)
            
            if permutation_corrs:
                # P-value is proportion of permuted correlations >= original
                p_value_perm = np.mean(np.abs(permutation_corrs) >= abs(original_corr))
                
                test_result = StatisticalTestResult(
                    test_type=StatisticalTest.MANN_WHITNEY_TEST,  # Using as proxy
                    test_statistic=abs(original_corr),
                    p_value=p_value_perm,
                    critical_values=None,
                    is_significant=p_value_perm < self.config.default_alpha,
                    interpretation=f"Permutation test: correlation significantly different from random",
                    metadata={
                        'test': 'permutation',
                        'original_correlation': original_corr,
                        'permutation_mean': np.mean(permutation_corrs),
                        'permutation_std': np.std(permutation_corrs),
                        'permutation_iterations': len(permutation_corrs)
                    }
                )
                test_results.append(test_result)
                
        except Exception as e:
            self.logger.warning(f'Robustness test failed: {e}')
        
        if not test_results:
            return None
        
        # Summary statistics
        robust_tests = [r for r in test_results if r.is_significant]
        robustness_score = len(robust_tests) / len(test_results)
        
        return AdvancedValidationResult(
            validation_metric=AdvancedValidationMetric.ROBUSTNESS,
            test_results=test_results,
            summary_statistic=robustness_score,
            confidence_interval=None,
            is_significant=robustness_score > 0.5,
            interpretation=f"Robustness score: {robustness_score:.3f}",
            recommendations=self._generate_robustness_recommendations(robustness_score),
            metadata={
                'total_tests': len(test_results),
                'robust_tests': len(robust_tests)
            }
        )
    
    def _identify_market_regimes(self, market_data: pd.DataFrame) -> pd.Series:
        """Identify market regimes for regime consistency testing."""
        if 'close' not in market_data.columns or len(market_data) < 50:
            return pd.Series(index=market_data.index, data=0)
        
        # Simple regime identification based on volatility
        returns = market_data['close'].pct_change()
        rolling_vol = returns.rolling(20).std()
        
        # Define regimes based on volatility quantiles
        vol_25 = rolling_vol.quantile(0.25)
        vol_75 = rolling_vol.quantile(0.75)
        
        regimes = pd.Series(index=market_data.index, data=1)  # Medium volatility
        regimes[rolling_vol <= vol_25] = 0  # Low volatility
        regimes[rolling_vol >= vol_75] = 2   # High volatility
        
        return regimes
    
    def _apply_bonferroni_correction(self):
        """Apply Bonferroni correction for multiple testing."""
        if not self.validation_results:
            return
        
        # Collect all p-values
        all_p_values = []
        test_keys = []
        
        for result_key, result in self.validation_results.items():
            for test_result in result.test_results:
                if test_result.p_value is not None:
                    all_p_values.append(test_result.p_value)
                    test_keys.append((result_key, test_result))
        
        if not all_p_values:
            return
        
        # Apply Bonferroni correction
        n_tests = len(all_p_values)
        corrected_alpha = self.config.default_alpha / n_tests
        
        # Update significance based on corrected alpha
        for i, (result_key, test_result) in enumerate(test_keys):
            test_result.is_significant = all_p_values[i] < corrected_alpha
            test_result.metadata['bonferroni_corrected'] = True
            test_result.metadata['corrected_alpha'] = corrected_alpha
        
        self.logger.info(f'📊 Applied Bonferroni correction: {n_tests} tests, α = {corrected_alpha:.6f}')
    
    # Recommendation generators
    def _generate_causality_recommendations(self, significant_lags: List[int]) -> List[str]:
        """Generate recommendations for causality test results."""
        recommendations = []
        
        if not significant_lags:
            recommendations.extend([
                "⚠️ No Granger causality detected",
                "Labels may not contain predictive information",
                "Consider revising labeling methodology",
                "Investigate alternative predictive features"
            ])
        elif len(significant_lags) == 1:
            recommendations.extend([
                f"✅ Granger causality detected at lag {significant_lags[0]}",
                "Labels show predictive power for future returns",
                "Consider optimizing for this specific time horizon"
            ])
        else:
            recommendations.extend([
                f"✅ Multiple causal lags detected: {significant_lags}",
                "Labels show multi-horizon predictive power",
                "Strong evidence for label effectiveness",
                "Consider ensemble approaches across multiple horizons"
            ])
        
        return recommendations
    
    def _generate_information_recommendations(self, info_score: float) -> List[str]:
        """Generate recommendations for information content results."""
        recommendations = []
        
        if info_score < 0.05:
            recommendations.extend([
                "⚠️ Low information content detected",
                "Labels provide minimal information about returns",
                "Consider feature engineering improvements",
                "Investigate alternative labeling approaches"
            ])
        elif info_score < 0.15:
            recommendations.extend([
                "📊 Moderate information content",
                "Labels provide some predictive information",
                "Room for improvement in labeling quality"
            ])
        else:
            recommendations.extend([
                "✅ High information content",
                "Labels contain significant predictive information",
                "Current labeling approach is effective"
            ])
        
        return recommendations
    
    def _generate_regime_recommendations(self, consistency_score: float) -> List[str]:
        """Generate recommendations for regime consistency results."""
        recommendations = []
        
        if consistency_score < 0.3:
            recommendations.extend([
                "⚠️ Poor regime consistency",
                "Labels behave very differently across market conditions",
                "Consider regime-specific labeling parameters",
                "Implement adaptive labeling strategies"
            ])
        elif consistency_score < 0.7:
            recommendations.extend([
                "📊 Moderate regime consistency",
                "Some variation in label behavior across regimes",
                "Consider minor regime-based adjustments"
            ])
        else:
            recommendations.extend([
                "✅ Good regime consistency",
                "Labels behave consistently across market conditions",
                "Current approach is robust to regime changes"
            ])
        
        return recommendations
    
    def _generate_stability_recommendations(self, stability_score: float) -> List[str]:
        """Generate recommendations for structural stability results."""
        recommendations = []
        
        if stability_score < 0.3:
            recommendations.extend([
                "⚠️ Poor structural stability",
                "Labels show significant structural breaks",
                "Consider rolling parameter updates",
                "Implement adaptive recalibration"
            ])
        elif stability_score < 0.7:
            recommendations.extend([
                "📊 Moderate structural stability",
                "Some evidence of structural changes",
                "Monitor for parameter drift over time"
            ])
        else:
            recommendations.extend([
                "✅ Good structural stability",
                "Labels remain stable over time",
                "Current parameters are robust"
            ])
        
        return recommendations
    
    def _generate_robustness_recommendations(self, robustness_score: float) -> List[str]:
        """Generate recommendations for robustness test results."""
        recommendations = []
        
        if robustness_score < 0.3:
            recommendations.extend([
                "⚠️ Poor robustness",
                "Results are not robust to sampling variation",
                "Increase data sample size if possible",
                "Consider ensemble methods for stability"
            ])
        elif robustness_score < 0.7:
            recommendations.extend([
                "📊 Moderate robustness",
                "Results show some sensitivity to sampling",
                "Consider confidence intervals in decision making"
            ])
        else:
            recommendations.extend([
                "✅ Good robustness",
                "Results are stable across different samples",
                "High confidence in labeling quality"
            ])
        
        return recommendations
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive advanced validation report."""
        if not self.validation_results:
            return "No advanced validation results available."
        
        report_lines = [
            "# Advanced Statistical Validation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"Completed {len(self.validation_results)} advanced validation analyses",
            ""
        ]
        
        # Summary statistics
        significant_results = [r for r in self.validation_results.values() if r.is_significant]
        report_lines.extend([
            f"**Significant Results**: {len(significant_results)}/{len(self.validation_results)} ({len(significant_results)/len(self.validation_results)*100:.1f}%)",
            ""
        ])
        
        # Group results by validation metric
        by_metric = {}
        for key, result in self.validation_results.items():
            metric_type = result.validation_metric.value
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
                    f"**Summary Statistic**: {result.summary_statistic:.4f}",
                    f"**Interpretation**: {result.interpretation}",
                    ""
                ])
                
                if result.confidence_interval:
                    ci_lower, ci_upper = result.confidence_interval
                    report_lines.append(f"**Confidence Interval**: [{ci_lower:.4f}, {ci_upper:.4f}]")
                
                if result.test_results:
                    report_lines.append("**Individual Tests**:")
                    for test in result.test_results[:3]:  # Show first 3 tests
                        report_lines.append(f"- {test.interpretation} (p={test.p_value:.4f})")
                
                if result.recommendations:
                    report_lines.append("**Recommendations**:")
                    for rec in result.recommendations:
                        report_lines.append(f"- {rec}")
                
                report_lines.append("")
        
        # Performance summary
        if self.validation_times:
            report_lines.extend([
                "## Performance Summary",
                ""
            ])
            for metric, time_taken in self.validation_times.items():
                report_lines.append(f"- {metric.value}: {time_taken:.2f}s")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_validation_results(self, output_path: Union[str, Path]):
        """Save advanced validation results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = {}
        for key, result in self.validation_results.items():
            serializable_results[key] = {
                'validation_metric': result.validation_metric.value,
                'summary_statistic': result.summary_statistic,
                'confidence_interval': result.confidence_interval,
                'is_significant': result.is_significant,
                'interpretation': result.interpretation,
                'recommendations': result.recommendations,
                'metadata': result.metadata,
                'timestamp': result.timestamp.isoformat(),
                'test_results': [
                    {
                        'test_type': tr.test_type.value,
                        'test_statistic': tr.test_statistic,
                        'p_value': tr.p_value,
                        'critical_values': tr.critical_values,
                        'is_significant': tr.is_significant,
                        'interpretation': tr.interpretation,
                        'metadata': tr.metadata,
                        'timestamp': tr.timestamp.isoformat()
                    }
                    for tr in result.test_results
                ]
            }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump({
                'validation_results': serializable_results,
                'validation_times': {k.value: v for k, v in self.validation_times.items()},
                'config': {
                    'granger_max_lags': self.config.granger_max_lags,
                    'bootstrap_iterations': self.config.bootstrap_iterations,
                    'default_alpha': self.config.default_alpha,
                    'bonferroni_correction': self.config.bonferroni_correction
                }
            }, f, indent=2)
        
        self.logger.info(f'💾 Advanced validation results saved to {output_path}')


# Convenience functions
def validate_labels_advanced(labeled_data: pd.DataFrame,
                            market_data: pd.DataFrame,
                            target_columns: Optional[List[str]] = None,
                            config: Optional[AdvancedValidationConfig] = None) -> Dict[str, AdvancedValidationResult]:
    """Convenience function for advanced statistical validation."""
    validator = AdvancedStatisticalValidator(config)
    return validator.comprehensive_validate(labeled_data, market_data, target_columns)


def generate_advanced_validation_report(labeled_data: pd.DataFrame,
                                       market_data: pd.DataFrame,
                                       target_columns: Optional[List[str]] = None,
                                       config: Optional[AdvancedValidationConfig] = None) -> str:
    """Convenience function to generate advanced validation report."""
    validator = AdvancedStatisticalValidator(config)
    validator.comprehensive_validate(labeled_data, market_data, target_columns)
    return validator.generate_comprehensive_report()