"""
Enhanced Statistical Analysis Framework with Advanced Features

This module provides comprehensive statistical analysis with hypothesis testing,
multiple testing corrections, and advanced statistical methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
from abc import ABC, abstractmethod
from scipy import stats
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mutual_info_score
import warnings

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import multiple testing corrections
try:
    from statsmodels.stats.multitest import multipletests
    MULTIPLE_TESTING_AVAILABLE = True
except ImportError:
    MULTIPLE_TESTING_AVAILABLE = False
    tprint_warning("⚠️ statsmodels not available, using fallback multiple testing corrections")

logger = logging.getLogger(__name__)

@dataclass
class HypothesisTestResult:
    """Result of a hypothesis test."""

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    degrees_of_freedom: Optional[int] = None
    alternative: str = "two-sided"
    alpha: float = 0.05

    def __post_init__(self):
        """Validate result."""
        assert 0 <= self.p_value <= 1, "p_value must be between 0 and 1"
        assert 0 < self.alpha < 1, "alpha must be between 0 and 1"

@dataclass
class MultipleTestingResult:
    """Result of multiple testing correction."""

    original_p_values: List[float]
    corrected_p_values: List[float]
    significant_indices: List[int]
    method: str
    alpha: float
    fdr_control: bool
    family_wise_error_rate: float

@dataclass
class StatisticalAnalysisResult:
    """Comprehensive statistical analysis result."""

    # Basic statistics
    n_samples: int
    n_features: int

    # Hypothesis tests
    hypothesis_tests: List[HypothesisTestResult]
    multiple_testing_result: Optional[MultipleTestingResult] = None

    # Correlation analysis
    correlation_matrix: Optional[pd.DataFrame] = None
    significant_correlations: List[Tuple[str, str, float, float]] = None

    # Mutual information
    mutual_information_matrix: Optional[pd.DataFrame] = None
    significant_mi: List[Tuple[str, str, float]] = None

    # Normality tests
    normality_tests: Dict[str, HypothesisTestResult] = None

    # Stationarity tests
    stationarity_tests: Dict[str, HypothesisTestResult] = None

    # Performance metrics
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0

class StatisticalTest(ABC):
    """Abstract base class for statistical tests."""

    def test(self, data: pd.DataFrame, **kwargs) -> HypothesisTestResult:
        """Perform the statistical test."""
        raise NotImplementedError(f"Subclasses must implement the test method. Class: {self.__class__.__name__}")

    def is_significant(self, result: HypothesisTestResult, alpha: float = 0.05) -> bool:
        """Check if the result is statistically significant."""
        raise NotImplementedError(f"Subclasses must implement the is_significant method. Class: {self.__class__.__name__}")

class NormalityTest(StatisticalTest):
    """Test for normality using multiple methods."""

    def test(self, data: pd.DataFrame, method: str = "shapiro", **kwargs) -> HypothesisTestResult:
        """Test normality using specified method."""
        results = []

        for col in data.columns:
            series = data[col].dropna()
            if len(series) < 3:
                continue

            if method == "shapiro":
                statistic, p_value = stats.shapiro(series)
            elif method == "anderson":
                result = stats.anderson(series, dist='norm')
                statistic = result.statistic
                p_value = result.critical_values[2]  # 5% significance level
            elif method == "ks":
                statistic, p_value = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
            else:
                raise ValueError(f"Unknown normality test method: {method}")

            results.append(HypothesisTestResult(
                test_name=f"{method}_normality_{col}",
                statistic=statistic,
                p_value=p_value,
                is_significant=p_value < kwargs.get('alpha', 0.05),
                alpha=kwargs.get('alpha', 0.05)
            ))

        # Return the most significant result
        if results:
            return max(results, key=lambda x: x.p_value)
        else:
            return HypothesisTestResult(
                test_name=f"{method}_normality",
                statistic=0.0,
                p_value=1.0,
                is_significant=False
            )

    def is_significant(self, result: HypothesisTestResult, alpha: float = 0.05) -> bool:
        """Check if data is significantly non-normal."""
        return result.p_value < alpha

class CorrelationTest(StatisticalTest):
    """Test for significant correlations."""

    def test(self, data: pd.DataFrame, method: str = "pearson", **kwargs) -> HypothesisTestResult:
        """Test for significant correlations between features."""
        if method == "pearson":
            corr_matrix = data.corr(method='pearson')
        elif method == "spearman":
            corr_matrix = data.corr(method='spearman')
        elif method == "kendall":
            corr_matrix = data.corr(method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        # Calculate p-values
        n = len(data)
        p_values = np.zeros_like(corr_matrix)

        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                if i != j:
                    try:
                        if method == "pearson":
                            _, p_val = pearsonr(data.iloc[:, i].dropna(), data.iloc[:, j].dropna())
                        elif method == "spearman":
                            _, p_val = spearmanr(data.iloc[:, i].dropna(), data.iloc[:, j].dropna())
                        elif method == "kendall":
                            _, p_val = kendalltau(data.iloc[:, i].dropna(), data.iloc[:, j].dropna())

                        p_values.iloc[i, j] = p_val
                    except Exception as e:
                        tprint_debug(f"Correlation test failed for {corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {e}")
                        p_values.iloc[i, j] = 1.0

        # Find most significant correlation
        max_corr = 0.0
        max_p_val = 1.0
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                p_val = p_values.iloc[i, j]

                if corr_val > max_corr:
                    max_corr = corr_val
                    max_p_val = p_val

        return HypothesisTestResult(
            test_name=f"{method}_correlation",
            statistic=max_corr,
            p_value=max_p_val,
            is_significant=max_p_val < kwargs.get('alpha', 0.05),
            alpha=kwargs.get('alpha', 0.05)
        )

    def is_significant(self, result: HypothesisTestResult, alpha: float = 0.05) -> bool:
        """Check if there are any significant correlations."""
        return result.p_value < alpha

class MutualInformationTest(StatisticalTest):
    """Test for mutual information between features."""

    def test(self, data: pd.DataFrame, targets: Optional[pd.Series] = None, **kwargs) -> HypothesisTestResult:
        """Calculate mutual information between features and targets."""
        if targets is None:
            # Calculate MI between all feature pairs
            mi_scores = []
            for i, col1 in enumerate(data.columns):
                for j, col2 in enumerate(data.columns):
                    if i != j:
                        try:
                            # Discretize continuous variables for MI calculation
                            data1 = pd.cut(data[col1].dropna(), bins=10, labels=False, duplicates='drop')
                            data2 = pd.cut(data[col2].dropna(), bins=10, labels=False, duplicates='drop')

                            # Align data
                            common_idx = data1.index.intersection(data2.index)
                            if len(common_idx) > 10:  # Minimum samples
                                mi = mutual_info_score(data1[common_idx], data2[common_idx])
                                mi_scores.append(mi)
                        except Exception as e:
                            tprint_debug(f"MI calculation failed for {col1} vs {col2}: {e}")

            max_mi = max(mi_scores) if mi_scores else 0.0
        else:
            # Calculate MI between features and targets
            mi_scores = []
            for col in data.columns:
                try:
                    if targets.dtype == 'object' or len(targets.unique()) < 10:
                        # Classification
                        mi = mutual_info_classif(data[[col]].dropna(), targets[data[col].dropna().index])
                    else:
                        # Regression
                        mi = mutual_info_regression(data[[col]].dropna(), targets[data[col].dropna().index])

                    mi_scores.append(mi[0] if hasattr(mi, '__len__') else mi)
                except Exception as e:
                    tprint_debug(f"MI calculation failed for {col}: {e}")
                    mi_scores.append(0.0)

            max_mi = max(mi_scores) if mi_scores else 0.0

        return HypothesisTestResult(
            test_name="mutual_information",
            statistic=max_mi,
            p_value=1.0 - max_mi,  # Convert to p-value approximation
            is_significant=max_mi > kwargs.get('threshold', 0.1),
            alpha=kwargs.get('alpha', 0.05)
        )

    def is_significant(self, result: HypothesisTestResult, alpha: float = 0.05) -> bool:
        """Check if there are significant mutual information values."""
        return result.statistic > 0.1  # Threshold for MI

class StationarityTest(StatisticalTest):
    """Test for stationarity using ADF test."""

    def test(self, data: pd.DataFrame, **kwargs) -> HypothesisTestResult:
        """Test stationarity using Augmented Dickey-Fuller test."""
        results = []

        for col in data.columns:
            series = data[col].dropna()
            if len(series) < 10:
                continue

            try:
                from statsmodels.tsa.stattools import adfuller
                statistic, p_value, _, _, critical_values, _ = adfuller(series, autolag='AIC')

                results.append(HypothesisTestResult(
                    test_name=f"adf_stationarity_{col}",
                    statistic=statistic,
                    p_value=p_value,
                    is_significant=p_value < kwargs.get('alpha', 0.05),
                    alpha=kwargs.get('alpha', 0.05)
                ))
            except ImportError:
                tprint_warning("statsmodels not available for ADF test")
                break
            except Exception as e:
                tprint_debug(f"ADF test failed for {col}: {e}")

        # Return the most significant result
        if results:
            return min(results, key=lambda x: x.p_value)
        else:
            return HypothesisTestResult(
                test_name="adf_stationarity",
                statistic=0.0,
                p_value=1.0,
                is_significant=False
            )

    def is_significant(self, result: HypothesisTestResult, alpha: float = 0.05) -> bool:
        """Check if data is stationary."""
        return result.p_value < alpha

class EnhancedStatisticalFramework:
    """
    Enhanced comprehensive statistical analysis framework.

    Provides advanced statistical analysis with hypothesis testing,
    multiple testing corrections, and comprehensive reporting.
    """

    def __init__(self, enable_multiple_testing: bool = True):
        """Initialize the enhanced statistical framework."""
        self.enable_multiple_testing = enable_multiple_testing and MULTIPLE_TESTING_AVAILABLE

        # Initialize test classes
        self.normality_test = NormalityTest()
        self.correlation_test = CorrelationTest()
        self.mi_test = MutualInformationTest()
        self.stationarity_test = StationarityTest()

        # Performance tracking
        self.performance_stats = {
            'total_tests_performed': 0,
            'significant_tests': 0,
            'multiple_testing_corrections': 0,
            'processing_time': 0.0,
            'memory_usage_mb': 0.0
        }

        tprint_info("Enhanced Statistical Framework initialized")
        if self.enable_multiple_testing:
            tprint_info("✅ Multiple testing corrections enabled")
        else:
            tprint_warning("⚠️ Multiple testing corrections disabled")

    def comprehensive_analysis(self, data: pd.DataFrame,
                             targets: Optional[pd.Series] = None,
                             alpha: float = 0.05,
                             multiple_testing_method: str = "fdr_bh") -> StatisticalAnalysisResult:
        """
        Perform comprehensive statistical analysis.

        Args:
            data: Input DataFrame
            targets: Optional target series
            alpha: Significance level
            multiple_testing_method: Method for multiple testing correction

        Returns:
            StatisticalAnalysisResult with comprehensive analysis
        """
        tprint_info(f"Starting comprehensive statistical analysis for {data.shape[0]} samples, {data.shape[1]} features")

        start_time = time.time()

        # Initialize result
        result = StatisticalAnalysisResult(
            n_samples=data.shape[0],
            n_features=data.shape[1],
            hypothesis_tests=[],
            processing_time=0.0
        )

        # Perform normality tests
        tprint_info("Performing normality tests...")
        normality_results = self._perform_normality_tests(data, alpha)
        result.hypothesis_tests.extend(normality_results)
        result.normality_tests = {r.test_name: r for r in normality_results}

        # Perform correlation tests
        tprint_info("Performing correlation analysis...")
        correlation_results = self._perform_correlation_tests(data, alpha)
        result.hypothesis_tests.extend(correlation_results)
        result.correlation_matrix = data.corr()
        result.significant_correlations = self._find_significant_correlations(data, alpha)

        # Perform mutual information tests
        tprint_info("Performing mutual information analysis...")
        mi_results = self._perform_mi_tests(data, targets, alpha)
        result.hypothesis_tests.extend(mi_results)
        result.mutual_information_matrix = self._calculate_mi_matrix(data, targets)
        result.significant_mi = self._find_significant_mi(data, targets, alpha)

        # Perform stationarity tests
        tprint_info("Performing stationarity tests...")
        stationarity_results = self._perform_stationarity_tests(data, alpha)
        result.hypothesis_tests.extend(stationarity_results)
        result.stationarity_tests = {r.test_name: r for r in stationarity_results}

        # Apply multiple testing correction
        if self.enable_multiple_testing and len(result.hypothesis_tests) > 1:
            tprint_info("Applying multiple testing correction...")
            result.multiple_testing_result = self._apply_multiple_testing_correction(
                result.hypothesis_tests, alpha, multiple_testing_method
            )
            self.performance_stats['multiple_testing_corrections'] += 1

        # Update performance stats
        total_time = time.time() - start_time
        result.processing_time = total_time
        self.performance_stats['total_tests_performed'] = len(result.hypothesis_tests)
        self.performance_stats['significant_tests'] = sum(1 for t in result.hypothesis_tests if t.is_significant)
        self.performance_stats['processing_time'] = total_time

        tprint_success(f"Comprehensive statistical analysis completed in {total_time:.3f}s")
        tprint_info(f"Performed {len(result.hypothesis_tests)} tests, {self.performance_stats['significant_tests']} significant")

        return result

    def _perform_normality_tests(self, data: pd.DataFrame, alpha: float) -> List[HypothesisTestResult]:
        """Perform normality tests on all columns."""
        results = []

        for col in data.columns:
            series = data[col].dropna()
            if len(series) < 3:
                continue

            # Shapiro-Wilk test
            try:
                shapiro_result = self.normality_test.test(data[[col]], method="shapiro", alpha=alpha)
                results.append(shapiro_result)
            except Exception as e:
                tprint_debug(f"Shapiro-Wilk test failed for {col}: {e}")

            # Anderson-Darling test
            try:
                ad_result = self.normality_test.test(data[[col]], method="anderson", alpha=alpha)
                results.append(ad_result)
            except Exception as e:
                tprint_debug(f"Anderson-Darling test failed for {col}: {e}")

        return results

    def _perform_correlation_tests(self, data: pd.DataFrame, alpha: float) -> List[HypothesisTestResult]:
        """Perform correlation tests."""
        results = []

        # Pearson correlation
        try:
            pearson_result = self.correlation_test.test(data, method="pearson", alpha=alpha)
            results.append(pearson_result)
        except Exception as e:
            tprint_debug(f"Pearson correlation test failed: {e}")

        # Spearman correlation
        try:
            spearman_result = self.correlation_test.test(data, method="spearman", alpha=alpha)
            results.append(spearman_result)
        except Exception as e:
            tprint_debug(f"Spearman correlation test failed: {e}")

        return results

    def _perform_mi_tests(self, data: pd.DataFrame, targets: Optional[pd.Series], alpha: float) -> List[HypothesisTestResult]:
        """Perform mutual information tests."""
        results = []

        try:
            mi_result = self.mi_test.test(data, targets, alpha=alpha)
            results.append(mi_result)
        except Exception as e:
            tprint_debug(f"Mutual information test failed: {e}")

        return results

    def _perform_stationarity_tests(self, data: pd.DataFrame, alpha: float) -> List[HypothesisTestResult]:
        """Perform stationarity tests."""
        results = []

        try:
            stationarity_result = self.stationarity_test.test(data, alpha=alpha)
            results.append(stationarity_result)
        except Exception as e:
            tprint_debug(f"Stationarity test failed: {e}")

        return results

    def _find_significant_correlations(self, data: pd.DataFrame, alpha: float) -> List[Tuple[str, str, float, float]]:
        """Find significant correlations."""
        significant_correlations = []

        try:
            corr_matrix = data.corr()
            n = len(data)

            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    try:
                        corr_val = corr_matrix.iloc[i, j]
                        _, p_val = pearsonr(data.iloc[:, i].dropna(), data.iloc[:, j].dropna())

                        if p_val < alpha:
                            significant_correlations.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_val,
                                p_val
                            ))
                    except Exception as e:
                        tprint_debug(f"Correlation calculation failed for {corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {e}")

        except Exception as e:
            tprint_debug(f"Significant correlation finding failed: {e}")

        return significant_correlations

    def _calculate_mi_matrix(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> pd.DataFrame:
        """Calculate mutual information matrix."""
        try:
            if targets is not None:
                # Calculate MI between features and targets
                mi_scores = {}
                for col in data.columns:
                    try:
                        if targets.dtype == 'object' or len(targets.unique()) < 10:
                            mi = mutual_info_classif(data[[col]].dropna(), targets[data[col].dropna().index])
                        else:
                            mi = mutual_info_regression(data[[col]].dropna(), targets[data[col].dropna().index])

                        mi_scores[col] = mi[0] if hasattr(mi, '__len__') else mi
                    except Exception as e:
                        tprint_debug(f"MI calculation failed for {col}: {e}")
                        mi_scores[col] = 0.0

                return pd.DataFrame([mi_scores], index=['target'])
            else:
                # Calculate MI between all feature pairs
                mi_matrix = np.zeros((len(data.columns), len(data.columns)))

                for i, col1 in enumerate(data.columns):
                    for j, col2 in enumerate(data.columns):
                        if i != j:
                            try:
                                data1 = pd.cut(data[col1].dropna(), bins=10, labels=False, duplicates='drop')
                                data2 = pd.cut(data[col2].dropna(), bins=10, labels=False, duplicates='drop')

                                common_idx = data1.index.intersection(data2.index)
                                if len(common_idx) > 10:
                                    mi = mutual_info_score(data1[common_idx], data2[common_idx])
                                    mi_matrix[i, j] = mi
                            except Exception as e:
                                tprint_debug(f"MI calculation failed for {col1} vs {col2}: {e}")

                return pd.DataFrame(mi_matrix, index=data.columns, columns=data.columns)

        except Exception as e:
            tprint_debug(f"MI matrix calculation failed: {e}")
            return pd.DataFrame()

    def _find_significant_mi(self, data: pd.DataFrame, targets: Optional[pd.Series], alpha: float) -> List[Tuple[str, str, float]]:
        """Find significant mutual information values."""
        significant_mi = []

        try:
            if targets is not None:
                # MI between features and targets
                for col in data.columns:
                    try:
                        if targets.dtype == 'object' or len(targets.unique()) < 10:
                            mi = mutual_info_classif(data[[col]].dropna(), targets[data[col].dropna().index])
                        else:
                            mi = mutual_info_regression(data[[col]].dropna(), targets[data[col].dropna().index])

                        mi_score = mi[0] if hasattr(mi, '__len__') else mi
                        if mi_score > 0.1:  # Threshold for significant MI
                            significant_mi.append((col, 'target', mi_score))
                    except Exception as e:
                        tprint_debug(f"MI calculation failed for {col}: {e}")
            else:
                # MI between feature pairs
                for i, col1 in enumerate(data.columns):
                    for j, col2 in enumerate(data.columns):
                        if i != j:
                            try:
                                data1 = pd.cut(data[col1].dropna(), bins=10, labels=False, duplicates='drop')
                                data2 = pd.cut(data[col2].dropna(), bins=10, labels=False, duplicates='drop')

                                common_idx = data1.index.intersection(data2.index)
                                if len(common_idx) > 10:
                                    mi = mutual_info_score(data1[common_idx], data2[common_idx])
                                    if mi > 0.1:
                                        significant_mi.append((col1, col2, mi))
                            except Exception as e:
                                tprint_debug(f"MI calculation failed for {col1} vs {col2}: {e}")

        except Exception as e:
            tprint_debug(f"Significant MI finding failed: {e}")

        return significant_mi

    def _apply_multiple_testing_correction(self, hypothesis_tests: List[HypothesisTestResult],
                                         alpha: float, method: str) -> MultipleTestingResult:
        """Apply multiple testing correction."""
        try:
            p_values = [test.p_value for test in hypothesis_tests]

            if MULTIPLE_TESTING_AVAILABLE:
                # Use statsmodels
                reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)

                significant_indices = [i for i, r in enumerate(reject) if r]
                fdr_control = method in ['fdr_bh', 'fdr_by']
                family_wise_error_rate = alpha
            else:
                # Fallback: Bonferroni correction
                corrected_p_values = [min(1.0, p * len(p_values)) for p in p_values]
                significant_indices = [i for i, p in enumerate(corrected_p_values) if p < alpha]
                fdr_control = False
                family_wise_error_rate = alpha / len(p_values)

            # Update hypothesis tests with corrected p-values
            for i, test in enumerate(hypothesis_tests):
                if MULTIPLE_TESTING_AVAILABLE:
                    test.p_value = pvals_corrected[i]
                    test.is_significant = reject[i]
                else:
                    test.p_value = corrected_p_values[i]
                    test.is_significant = corrected_p_values[i] < alpha

            return MultipleTestingResult(
                original_p_values=p_values,
                corrected_p_values=pvals_corrected if MULTIPLE_TESTING_AVAILABLE else corrected_p_values,
                significant_indices=significant_indices,
                method=method,
                alpha=alpha,
                fdr_control=fdr_control,
                family_wise_error_rate=family_wise_error_rate
            )

        except Exception as e:
            tprint_error(f"Multiple testing correction failed: {e}")
            return MultipleTestingResult(
                original_p_values=[test.p_value for test in hypothesis_tests],
                corrected_p_values=[test.p_value for test in hypothesis_tests],
                significant_indices=[],
                method=method,
                alpha=alpha,
                fdr_control=False,
                family_wise_error_rate=alpha
            )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.performance_stats.copy()

# Convenience functions
def create_enhanced_statistical_framework(enable_multiple_testing: bool = True) -> EnhancedStatisticalFramework:
    """Create an enhanced statistical framework."""
    return EnhancedStatisticalFramework(enable_multiple_testing=enable_multiple_testing)

def perform_comprehensive_analysis(data: pd.DataFrame,
                                 targets: Optional[pd.Series] = None,
                                 alpha: float = 0.05,
                                 multiple_testing_method: str = "fdr_bh") -> StatisticalAnalysisResult:
    """Perform comprehensive statistical analysis."""
    framework = create_enhanced_statistical_framework()
    return framework.comprehensive_analysis(data, targets, alpha, multiple_testing_method)
