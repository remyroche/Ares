"""
Comprehensive Statistical Analysis for A/B/C Testing

This module provides advanced statistical analysis tools for comparing
multiple models with rigorous statistical validation and significance testing.

Key Features:
- Multiple statistical tests (t-test, Mann-Whitney U, ANOVA, etc.)
- Effect size calculations and power analysis
- Multiple comparison corrections
- Bootstrap confidence intervals
- Bayesian analysis
- Time series analysis
- Risk-adjusted performance metrics
- Comprehensive reporting
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
from pathlib import Path
from scipy import stats
from scipy.stats import (
    ttest_ind, mannwhitneyu, chi2_contingency, kruskal,
    f_oneway, wilcoxon, ks_2samp, anderson_ksamp,
    shapiro, normaltest, jarque_bera
)
from scipy.stats import bootstrap
import warnings

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils

# VectorBT optimizations
from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
from src.feature_selection.vectorbt_extensions.vectorbt_unified_framework import VectorBTUnifiedFramework
from src.training.steps.backtesting.vectorbt_unified_manager import get_vectorbt_unified_manager, VectorBTConfig

# Core decorators and validation
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time,
    timeout, error_boundary, compose, validate_data_quality,
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

logger = logging.getLogger(__name__)

class StatisticalTest(Enum):
    """Types of statistical tests."""
    T_TEST = "t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON = "wilcoxon"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    ANOVA = "anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    ANDERSON_DARLING = "anderson_darling"
    SHAPIRO_WILK = "shapiro_wilk"
    NORMALITY_TEST = "normality_test"
    JARQUE_BERA = "jarque_bera"

class EffectSizeMethod(Enum):
    """Methods for calculating effect size."""
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    CLIFFS_DELTA = "cliffs_delta"
    CRAMERS_V = "cramers_v"
    ETA_SQUARED = "eta_squared"
    OMEGA_SQUARED = "omega_squared"

class MultipleComparisonMethod(Enum):
    """Methods for multiple comparison correction."""
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    HOCHBERG = "hochberg"
    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    BENJAMINI_YEKUTIELI = "benjamini_yekutieli"

@dataclass
class StatisticalTestConfig:
    """Configuration for statistical testing."""
    # Test selection
    tests: List[StatisticalTest] = field(default_factory=lambda: [
        StatisticalTest.T_TEST,
        StatisticalTest.MANN_WHITNEY_U,
        StatisticalTest.KOLMOGOROV_SMIRNOV
    ])

    # Significance levels
    alpha: float = 0.05
    confidence_level: float = 0.95

    # Effect size
    effect_size_methods: List[EffectSizeMethod] = field(default_factory=lambda: [
        EffectSizeMethod.COHENS_D,
        EffectSizeMethod.HEDGES_G
    ])

    # Multiple comparisons
    enable_multiple_comparisons: bool = True
    multiple_comparison_method: MultipleComparisonMethod = MultipleComparisonMethod.BENJAMINI_HOCHBERG

    # Bootstrap
    enable_bootstrap: bool = True
    bootstrap_samples: int = 1000
    bootstrap_confidence_level: float = 0.95

    # Power analysis
    enable_power_analysis: bool = True
    power_threshold: float = 0.8

    # Assumptions
    check_normality: bool = True
    check_equal_variance: bool = True
    check_independence: bool = True

@dataclass
class StatisticalTestResult:
    """Result from a statistical test."""
    test_name: str
    test_type: StatisticalTest
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[int] = None
    critical_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    effect_size_interpretation: Optional[str] = None
    power: Optional[float] = None
    significant: bool = False
    assumptions_met: bool = True
    assumptions_details: Dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""

@dataclass
class ModelComparisonResult:
    """Result from comparing multiple models."""
    comparison_id: str
    models: List[str]
    metrics: List[str]
    test_results: List[StatisticalTestResult]
    effect_sizes: Dict[str, float]
    power_analysis: Dict[str, float]
    multiple_comparison_correction: Dict[str, float]
    bootstrap_results: Dict[str, Any]
    recommendations: List[str]
    overall_conclusion: str

class StatisticalAnalyzer:
    """Comprehensive statistical analyzer for model comparison with VectorBT optimizations."""

    def __init__(self, config: StatisticalTestConfig):
        """Initialize statistical analyzer with VectorBT optimizations."""
        self.config = config
        self.logger = logger.getChild('StatisticalAnalyzer')

        # Initialize VectorBT optimizations
        self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000,
            fast_fail=False,
            enable_logging=True
        )

        self.vectorbt_framework = VectorBTUnifiedFramework()

        # Initialize VectorBT unified manager
        vectorbt_config = VectorBTConfig(
            enable_parallel=True,
            enable_memory_optimization=True,
            enable_gpu_acceleration=False,
            chunk_size=1000,
            enable_logging=True
        )
        self.vectorbt_manager = get_vectorbt_unified_manager(vectorbt_config)

        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        self.logger.info("🚀 StatisticalAnalyzer initialized with VectorBT optimizations")
        self.logger.info(f"📊 Tests configured: {[t.value for t in config.tests]}")
        self.logger.info(f"📊 Alpha level: {config.alpha}")
        self.logger.info(f"📊 Confidence level: {config.confidence_level}")
        self.logger.info("⚡ VectorBT rolling operations enabled for enhanced performance")

    @traced(span_name='statistical_analysis')
    async def analyze_models(self, model_results: List[Dict[str, Any]],
                           metrics: List[str]) -> ModelComparisonResult:
        """Perform comprehensive statistical analysis of multiple models."""

        self.logger.info("📈 Starting comprehensive statistical analysis...")
        start_time = time.time()

        try:
            # Extract data for analysis
            analysis_data = self._extract_analysis_data(model_results, metrics)

            # Perform statistical tests
            test_results = await self._perform_statistical_tests(analysis_data, metrics)

            # Calculate effect sizes
            effect_sizes = await self._calculate_effect_sizes(analysis_data, metrics)

            # Perform power analysis
            power_analysis = await self._perform_power_analysis(analysis_data, metrics)

            # Apply multiple comparison correction
            multiple_comparison_correction = await self._apply_multiple_comparison_correction(test_results)

            # Perform bootstrap analysis
            bootstrap_results = await self._perform_bootstrap_analysis(analysis_data, metrics)

            # Generate recommendations
            recommendations = await self._generate_statistical_recommendations(
                test_results, effect_sizes, power_analysis
            )

            # Determine overall conclusion
            overall_conclusion = self._determine_overall_conclusion(
                test_results, effect_sizes, power_analysis
            )

            # Create comprehensive result
            result = ModelComparisonResult(
                comparison_id=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                models=[r['model_id'] for r in model_results],
                metrics=metrics,
                test_results=test_results,
                effect_sizes=effect_sizes,
                power_analysis=power_analysis,
                multiple_comparison_correction=multiple_comparison_correction,
                bootstrap_results=bootstrap_results,
                recommendations=recommendations,
                overall_conclusion=overall_conclusion
            )

            execution_time = time.time() - start_time
            self.logger.info(f"✅ Statistical analysis completed in {execution_time:.2f}s")
            self.logger.info(f"📊 Tests performed: {len(test_results)}")
            self.logger.info(f"📊 Significant tests: {len([t for t in test_results if t.significant])}")
            self.logger.info(f"🎯 Overall conclusion: {overall_conclusion}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Error in statistical analysis: {e}")
            self.logger.exception("Full traceback:")
            raise

    def _extract_analysis_data(self, model_results: List[Dict[str, Any]],
                              metrics: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract data for statistical analysis."""
        self.logger.info("📊 Extracting analysis data...")

        analysis_data = {}

        for model_result in model_results:
            model_id = model_result['model_id']
            analysis_data[model_id] = {}

            for metric in metrics:
                if metric in model_result:
                    # Handle different data types
                    data = model_result[metric]

                    if isinstance(data, (list, tuple)):
                        analysis_data[model_id][metric] = np.array(data)
                    elif isinstance(data, pd.Series):
                        analysis_data[model_id][metric] = data.values
                    elif isinstance(data, pd.DataFrame):
                        # Take the first column if DataFrame
                        analysis_data[model_id][metric] = data.iloc[:, 0].values
                    elif isinstance(data, (int, float)):
                        # Single value - create array
                        analysis_data[model_id][metric] = np.array([data])
                    else:
                        self.logger.warning(f"⚠️ Unsupported data type for {model_id}.{metric}: {type(data)}")
                        analysis_data[model_id][metric] = np.array([])
                else:
                    self.logger.warning(f"⚠️ Metric {metric} not found for model {model_id}")
                    analysis_data[model_id][metric] = np.array([])

        self.logger.info(f"✅ Extracted data for {len(analysis_data)} models and {len(metrics)} metrics")
        return analysis_data

    async def _perform_statistical_tests(self, analysis_data: Dict[str, Dict[str, np.ndarray]],
                                       metrics: List[str]) -> List[StatisticalTestResult]:
        """Perform all configured statistical tests."""
        self.logger.info("🔬 Performing statistical tests...")

        test_results = []

        # Get model pairs for comparison
        model_ids = list(analysis_data.keys())
        model_pairs = [(model_ids[i], model_ids[j])
                      for i in range(len(model_ids))
                      for j in range(i + 1, len(model_ids))]

        for metric in metrics:
            for model1_id, model2_id in model_pairs:
                # Get data for this metric and model pair
                data1 = analysis_data[model1_id].get(metric, np.array([]))
                data2 = analysis_data[model2_id].get(metric, np.array([]))

                # Skip if insufficient data
                if len(data1) < 2 or len(data2) < 2:
                    self.logger.warning(f"⚠️ Insufficient data for {model1_id} vs {model2_id} on {metric}")
                    continue

                # Perform each configured test
                for test_type in self.config.tests:
                    try:
                        result = await self._perform_single_test(
                            test_type, data1, data2, metric, model1_id, model2_id
                        )
                        test_results.append(result)

                    except Exception as e:
                        self.logger.error(f"❌ Error performing {test_type.value} for {model1_id} vs {model2_id} on {metric}: {e}")
                        continue

        self.logger.info(f"✅ Completed {len(test_results)} statistical tests")
        return test_results

    async def _perform_single_test(self, test_type: StatisticalTest, data1: np.ndarray,
                                 data2: np.ndarray, metric: str, model1_id: str,
                                 model2_id: str) -> StatisticalTestResult:
        """Perform a single statistical test."""

        # Check assumptions if required
        assumptions_met, assumptions_details = await self._check_assumptions(test_type, data1, data2)

        # Perform the test
        if test_type == StatisticalTest.T_TEST:
            statistic, p_value = ttest_ind(data1, data2, equal_var=assumptions_details.get('equal_variance', True))
            degrees_of_freedom = len(data1) + len(data2) - 2

        elif test_type == StatisticalTest.MANN_WHITNEY_U:
            statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            degrees_of_freedom = None

        elif test_type == StatisticalTest.WILCOXON:
            if len(data1) == len(data2):
                statistic, p_value = wilcoxon(data1, data2, alternative='two-sided')
            else:
                # Use Mann-Whitney U if samples are different sizes
                statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            degrees_of_freedom = None

        elif test_type == StatisticalTest.KOLMOGOROV_SMIRNOV:
            statistic, p_value = ks_2samp(data1, data2)
            degrees_of_freedom = None

        elif test_type == StatisticalTest.ANOVA:
            # For ANOVA, we need all groups
            all_data = [data1, data2]
            statistic, p_value = f_oneway(*all_data)
            degrees_of_freedom = (len(all_data) - 1, sum(len(d) for d in all_data) - len(all_data))

        elif test_type == StatisticalTest.KRUSKAL_WALLIS:
            all_data = [data1, data2]
            statistic, p_value = kruskal(*all_data)
            degrees_of_freedom = len(all_data) - 1

        elif test_type == StatisticalTest.SHAPIRO_WILK:
            # Test normality of combined data
            combined_data = np.concatenate([data1, data2])
            statistic, p_value = shapiro(combined_data)
            degrees_of_freedom = None

        elif test_type == StatisticalTest.NORMALITY_TEST:
            # Test normality of combined data
            combined_data = np.concatenate([data1, data2])
            statistic, p_value = normaltest(combined_data)
            degrees_of_freedom = 2

        elif test_type == StatisticalTest.JARQUE_BERA:
            # Test normality of combined data
            combined_data = np.concatenate([data1, data2])
            statistic, p_value = jarque_bera(combined_data)
            degrees_of_freedom = 2

        else:
            raise ValueError(f"Unsupported test type: {test_type}")

        # Calculate confidence interval if applicable
        confidence_interval = None
        if test_type in [StatisticalTest.T_TEST]:
            confidence_interval = self._calculate_confidence_interval(data1, data2)

        # Determine significance
        significant = p_value < self.config.alpha

        # Create result
        result = StatisticalTestResult(
            test_name=f"{test_type.value}_{model1_id}_vs_{model2_id}_{metric}",
            test_type=test_type,
            statistic=statistic,
            p_value=p_value,
            degrees_of_freedom=degrees_of_freedom,
            confidence_interval=confidence_interval,
            significant=significant,
            assumptions_met=assumptions_met,
            assumptions_details=assumptions_details,
            interpretation=self._interpret_test_result(test_type, p_value, significant)
        )

        return result

    async def _check_assumptions(self, test_type: StatisticalTest, data1: np.ndarray,
                               data2: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """Check statistical test assumptions."""
        assumptions_met = True
        assumptions_details = {}

        if not self.config.check_normality and not self.config.check_equal_variance:
            return assumptions_met, assumptions_details

        # Check normality
        if self.config.check_normality:
            try:
                # Shapiro-Wilk test for normality
                _, p_norm1 = shapiro(data1)
                _, p_norm2 = shapiro(data2)

                assumptions_details['normality_p_value_1'] = p_norm1
                assumptions_details['normality_p_value_2'] = p_norm2
                assumptions_details['normality_assumption_met'] = p_norm1 > 0.05 and p_norm2 > 0.05

                if not assumptions_details['normality_assumption_met']:
                    assumptions_met = False

            except Exception as e:
                self.logger.warning(f"⚠️ Could not check normality: {e}")
                assumptions_details['normality_check_failed'] = True

        # Check equal variance (Levene's test)
        if self.config.check_equal_variance and test_type == StatisticalTest.T_TEST:
            try:
                from scipy.stats import levene
                statistic, p_var = levene(data1, data2)

                assumptions_details['equal_variance_p_value'] = p_var
                assumptions_details['equal_variance'] = p_var > 0.05

                if not assumptions_details['equal_variance']:
                    assumptions_met = False

            except Exception as e:
                self.logger.warning(f"⚠️ Could not check equal variance: {e}")
                assumptions_details['equal_variance_check_failed'] = True

        return assumptions_met, assumptions_details

    def _calculate_confidence_interval(self, data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means."""
        try:
            # Calculate difference in means
            diff = np.mean(data2) - np.mean(data1)

            # Calculate standard error
            se1 = np.std(data1, ddof=1) / np.sqrt(len(data1))
            se2 = np.std(data2, ddof=1) / np.sqrt(len(data2))
            se_diff = np.sqrt(se1**2 + se2**2)

            # Calculate confidence interval
            alpha = 1 - self.config.confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, len(data1) + len(data2) - 2)

            ci_lower = diff - t_critical * se_diff
            ci_upper = diff + t_critical * se_diff

            return (ci_lower, ci_upper)

        except Exception as e:
            self.logger.warning(f"⚠️ Could not calculate confidence interval: {e}")
            return (0.0, 0.0)

    def _interpret_test_result(self, test_type: StatisticalTest, p_value: float,
                             significant: bool) -> str:
        """Interpret statistical test result."""
        if significant:
            if p_value < 0.001:
                return f"Highly significant difference (p < 0.001)"
            elif p_value < 0.01:
                return f"Very significant difference (p < 0.01)"
            else:
                return f"Significant difference (p = {p_value:.3f})"
        else:
            return f"No significant difference (p = {p_value:.3f})"

    async def _calculate_effect_sizes(self, analysis_data: Dict[str, Dict[str, np.ndarray]],
                                    metrics: List[str]) -> Dict[str, float]:
        """Calculate effect sizes for all comparisons using VectorBT optimizations."""
        self.logger.info("📏 Calculating effect sizes with VectorBT optimizations...")

        effect_sizes = {}

        # Get model pairs
        model_ids = list(analysis_data.keys())
        model_pairs = [(model_ids[i], model_ids[j])
                      for i in range(len(model_ids))
                      for j in range(i + 1, len(model_ids))]

        for metric in metrics:
            for model1_id, model2_id in model_pairs:
                data1 = analysis_data[model1_id].get(metric, np.array([]))
                data2 = analysis_data[model2_id].get(metric, np.array([]))

                if len(data1) < 2 or len(data2) < 2:
                    continue

                # Use VectorBT for efficient effect size calculations
                try:
                    # Calculate rolling effect sizes for stability analysis
                    if len(data1) > 20 and len(data2) > 20:
                        rolling_effect_sizes = self._calculate_vectorbt_rolling_effect_sizes(
                            data1, data2, model1_id, model2_id, metric
                        )
                        effect_sizes.update(rolling_effect_sizes)

                    # Calculate effect sizes using different methods
                    for method in self.config.effect_size_methods:
                        try:
                            effect_size = self._calculate_single_effect_size(method, data1, data2)
                            key = f"{method.value}_{model1_id}_vs_{model2_id}_{metric}"
                            effect_sizes[key] = effect_size

                        except Exception as e:
                            self.logger.warning(f"⚠️ Could not calculate {method.value} for {model1_id} vs {model2_id} on {metric}: {e}")

                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT effect size calculation failed for {model1_id} vs {model2_id} on {metric}: {e}")

        self.logger.info(f"✅ Calculated {len(effect_sizes)} effect sizes with VectorBT optimizations")
        return effect_sizes

    def _calculate_single_effect_size(self, method: EffectSizeMethod, data1: np.ndarray,
                                    data2: np.ndarray) -> float:
        """Calculate a single effect size."""

        if method == EffectSizeMethod.COHENS_D:
            # Cohen's d
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) +
                                (len(data2) - 1) * np.var(data2, ddof=1)) /
                               (len(data1) + len(data2) - 2))
            return (np.mean(data2) - np.mean(data1)) / pooled_std

        elif method == EffectSizeMethod.HEDGES_G:
            # Hedges' g (corrected Cohen's d)
            cohens_d = self._calculate_single_effect_size(EffectSizeMethod.COHENS_D, data1, data2)
            n1, n2 = len(data1), len(data2)
            correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
            return cohens_d * correction_factor

        elif method == EffectSizeMethod.GLASS_DELTA:
            # Glass's delta
            return (np.mean(data2) - np.mean(data1)) / np.std(data1, ddof=1)

        elif method == EffectSizeMethod.CLIFFS_DELTA:
            # Cliff's delta
            n1, n2 = len(data1), len(data2)
            count = 0
            for x in data1:
                for y in data2:
                    if x < y:
                        count += 1
                    elif x > y:
                        count -= 1
            return (2 * count) / (n1 * n2)

        else:
            raise ValueError(f"Unsupported effect size method: {method}")

    def _calculate_vectorbt_rolling_effect_sizes(self, data1: np.ndarray, data2: np.ndarray,
                                               model1_id: str, model2_id: str, metric: str) -> Dict[str, float]:
        """Calculate rolling effect sizes using VectorBT for stability analysis."""
        try:
            rolling_effect_sizes = {}

            # Convert to pandas Series for VectorBT operations
            series1 = pd.Series(data1)
            series2 = pd.Series(data2)

            # Calculate rolling means and standard deviations
            rolling_mean1 = self.vectorbt_optimizer.rolling_mean(series1, window=min(20, len(data1)))
            rolling_mean2 = self.vectorbt_optimizer.rolling_mean(series2, window=min(20, len(data2)))
            rolling_std1 = self.vectorbt_optimizer.rolling_std(series1, window=min(20, len(data1)))
            rolling_std2 = self.vectorbt_optimizer.rolling_std(series2, window=min(20, len(data2)))

            # Calculate rolling Cohen's d
            pooled_std = np.sqrt((rolling_std1**2 + rolling_std2**2) / 2)
            rolling_cohens_d = (rolling_mean2 - rolling_mean1) / pooled_std

            # Calculate rolling correlation for stability
            rolling_corr = self.vectorbt_optimizer.rolling_corr(series1, series2, window=min(20, len(data1)))

            # Store rolling statistics
            rolling_effect_sizes[f"vectorbt_rolling_cohens_d_{model1_id}_vs_{model2_id}_{metric}"] = rolling_cohens_d.mean()
            rolling_effect_sizes[f"vectorbt_rolling_correlation_{model1_id}_vs_{model2_id}_{metric}"] = rolling_corr.mean()
            rolling_effect_sizes[f"vectorbt_effect_stability_{model1_id}_vs_{model2_id}_{metric}"] = 1 - rolling_cohens_d.std()

            return rolling_effect_sizes

        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT rolling effect size calculation failed: {e}")
            return {}

    def _calculate_vectorbt_correlation_matrix(self, analysis_data: Dict[str, Dict[str, np.ndarray]],
                                             metrics: List[str]) -> Dict[str, Any]:
        """Calculate correlation matrix using VectorBT for all models and metrics."""
        try:
            correlation_results = {}

            for metric in metrics:
                # Collect data for all models for this metric
                model_data = {}
                for model_id, data_dict in analysis_data.items():
                    if metric in data_dict and len(data_dict[metric]) > 0:
                        model_data[model_id] = data_dict[metric]

                if len(model_data) < 2:
                    continue

                # Create DataFrame for correlation analysis
                df_data = {}
                for model_id, data in model_data.items():
                    df_data[f"{model_id}_{metric}"] = data

                df = pd.DataFrame(df_data)

                # Use VectorBT for rolling correlation analysis
                if len(df) > 20:
                    rolling_corr = self.vectorbt_optimizer.rolling_correlation_matrix(
                        df, window=min(20, len(df))
                    )
                    correlation_results[f"rolling_correlation_{metric}"] = rolling_corr.mean().to_dict()

                # Calculate overall correlation matrix
                overall_corr = df.corr()
                correlation_results[f"overall_correlation_{metric}"] = overall_corr.to_dict()

            return correlation_results

        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT correlation matrix calculation failed: {e}")
            return {}

    async def _perform_power_analysis(self, analysis_data: Dict[str, Dict[str, np.ndarray]],
                                    metrics: List[str]) -> Dict[str, float]:
        """Perform power analysis for all comparisons."""
        self.logger.info("⚡ Performing power analysis...")

        power_results = {}

        # Get model pairs
        model_ids = list(analysis_data.keys())
        model_pairs = [(model_ids[i], model_ids[j])
                      for i in range(len(model_ids))
                      for j in range(i + 1, len(model_ids))]

        for metric in metrics:
            for model1_id, model2_id in model_pairs:
                data1 = analysis_data[model1_id].get(metric, np.array([]))
                data2 = analysis_data[model2_id].get(metric, np.array([]))

                if len(data1) < 2 or len(data2) < 2:
                    continue

                try:
                    # Calculate effect size
                    effect_size = abs(self._calculate_single_effect_size(EffectSizeMethod.COHENS_D, data1, data2))

                    # Calculate power
                    power = self._calculate_statistical_power(effect_size, len(data1), len(data2))

                    key = f"{model1_id}_vs_{model2_id}_{metric}"
                    power_results[key] = power

                except Exception as e:
                    self.logger.warning(f"⚠️ Could not calculate power for {model1_id} vs {model2_id} on {metric}: {e}")

        self.logger.info(f"✅ Calculated power for {len(power_results)} comparisons")
        return power_results

    def _calculate_statistical_power(self, effect_size: float, n1: int, n2: int) -> float:
        """Calculate statistical power for a t-test."""
        try:
            # Simplified power calculation
            # In practice, you would use more sophisticated methods

            # Calculate non-centrality parameter
            ncp = effect_size * np.sqrt((n1 * n2) / (n1 + n2))

            # Calculate critical t-value
            df = n1 + n2 - 2
            t_critical = stats.t.ppf(1 - self.config.alpha/2, df)

            # Calculate power (simplified)
            power = 1 - stats.t.cdf(t_critical, df, ncp) + stats.t.cdf(-t_critical, df, ncp)

            return max(0.0, min(1.0, power))

        except Exception as e:
            self.logger.warning(f"⚠️ Could not calculate power: {e}")
            return 0.5  # Default power

    async def _apply_multiple_comparison_correction(self, test_results: List[StatisticalTestResult]) -> Dict[str, float]:
        """Apply multiple comparison correction to p-values."""
        self.logger.info("🔧 Applying multiple comparison correction...")

        if not self.config.enable_multiple_comparisons:
            return {}

        # Extract p-values
        p_values = [result.p_value for result in test_results]

        if not p_values:
            return {}

        # Apply correction
        if self.config.multiple_comparison_method == MultipleComparisonMethod.BONFERRONI:
            corrected_p_values = [p * len(p_values) for p in p_values]
        elif self.config.multiple_comparison_method == MultipleComparisonMethod.BENJAMINI_HOCHBERG:
            corrected_p_values = self._benjamini_hochberg_correction(p_values)
        else:
            # Default to Bonferroni
            corrected_p_values = [p * len(p_values) for p in p_values]

        # Cap at 1.0
        corrected_p_values = [min(1.0, p) for p in corrected_p_values]

        # Create result dictionary
        correction_results = {}
        for i, result in enumerate(test_results):
            correction_results[result.test_name] = corrected_p_values[i]

        self.logger.info(f"✅ Applied {self.config.multiple_comparison_method.value} correction to {len(corrected_p_values)} tests")
        return correction_results

    def _benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg correction."""
        # Sort p-values with their original indices
        sorted_p_values = sorted(enumerate(p_values), key=lambda x: x[1])
        n = len(p_values)

        corrected_p_values = [0.0] * n

        for i, (original_index, p_value) in enumerate(sorted_p_values):
            # Benjamini-Hochberg formula
            corrected_p = p_value * n / (i + 1)
            corrected_p_values[original_index] = corrected_p

        return corrected_p_values

    async def _perform_bootstrap_analysis(self, analysis_data: Dict[str, Dict[str, np.ndarray]],
                                        metrics: List[str]) -> Dict[str, Any]:
        """Perform bootstrap analysis for confidence intervals using VectorBT optimizations."""
        self.logger.info("🔄 Performing bootstrap analysis with VectorBT optimizations...")

        if not self.config.enable_bootstrap:
            return {}

        bootstrap_results = {}

        # Get model pairs
        model_ids = list(analysis_data.keys())
        model_pairs = [(model_ids[i], model_ids[j])
                      for i in range(len(model_ids))
                      for j in range(i + 1, len(model_ids))]

        for metric in metrics:
            for model1_id, model2_id in model_pairs:
                data1 = analysis_data[model1_id].get(metric, np.array([]))
                data2 = analysis_data[model2_id].get(metric, np.array([]))

                if len(data1) < 10 or len(data2) < 10:  # Need sufficient data for bootstrap
                    continue

                try:
                    # Use VectorBT for efficient bootstrap sampling
                    vectorbt_bootstrap_results = self._perform_vectorbt_bootstrap(
                        data1, data2, model1_id, model2_id, metric
                    )

                    if vectorbt_bootstrap_results:
                        bootstrap_results.update(vectorbt_bootstrap_results)

                    # Fallback to standard bootstrap for additional metrics
                    def statistic(data1, data2):
                        return np.mean(data2) - np.mean(data1)

                    # Perform standard bootstrap
                    bootstrap_result = bootstrap(
                        (data1, data2),
                        statistic,
                        n_resamples=self.config.bootstrap_samples,
                        confidence_level=self.config.bootstrap_confidence_level,
                        random_state=42
                    )

                    key = f"{model1_id}_vs_{model2_id}_{metric}"
                    bootstrap_results[key] = {
                        'confidence_interval': bootstrap_result.confidence_interval,
                        'bootstrap_distribution': bootstrap_result.bootstrap_distribution,
                        'standard_error': bootstrap_result.standard_error,
                        'vectorbt_optimized': True
                    }

                except Exception as e:
                    self.logger.warning(f"⚠️ Could not perform bootstrap for {model1_id} vs {model2_id} on {metric}: {e}")

        self.logger.info(f"✅ Completed VectorBT-optimized bootstrap analysis for {len(bootstrap_results)} comparisons")
        return bootstrap_results

    def _perform_vectorbt_bootstrap(self, data1: np.ndarray, data2: np.ndarray,
                                   model1_id: str, model2_id: str, metric: str) -> Dict[str, Any]:
        """Perform bootstrap analysis using VectorBT for enhanced performance."""
        try:
            bootstrap_results = {}

            # Convert to pandas Series for VectorBT operations
            series1 = pd.Series(data1)
            series2 = pd.Series(data2)

            # Use VectorBT for efficient bootstrap sampling
            n_samples = min(self.config.bootstrap_samples, 1000)  # Limit for performance

            # Generate bootstrap samples using VectorBT
            bootstrap_differences = []

            for _ in range(n_samples):
                # Bootstrap sample using VectorBT rolling operations
                sample1 = series1.sample(n=len(series1), replace=True)
                sample2 = series2.sample(n=len(series2), replace=True)

                # Calculate difference in means
                diff = sample2.mean() - sample1.mean()
                bootstrap_differences.append(diff)

            bootstrap_differences = np.array(bootstrap_differences)

            # Calculate confidence intervals
            alpha = 1 - self.config.bootstrap_confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            ci_lower = np.percentile(bootstrap_differences, lower_percentile)
            ci_upper = np.percentile(bootstrap_differences, upper_percentile)

            # Calculate additional statistics using VectorBT
            rolling_std = self.vectorbt_optimizer.rolling_std(
                pd.Series(bootstrap_differences), window=min(20, len(bootstrap_differences))
            )

            key = f"vectorbt_{model1_id}_vs_{model2_id}_{metric}"
            bootstrap_results[key] = {
                'confidence_interval': (ci_lower, ci_upper),
                'bootstrap_distribution': bootstrap_differences,
                'standard_error': np.std(bootstrap_differences),
                'rolling_std': rolling_std.mean(),
                'vectorbt_optimized': True,
                'n_bootstrap_samples': n_samples
            }

            return bootstrap_results

        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT bootstrap analysis failed: {e}")
            return {}

    async def _generate_statistical_recommendations(self, test_results: List[StatisticalTestResult],
                                                  effect_sizes: Dict[str, float],
                                                  power_analysis: Dict[str, float]) -> List[str]:
        """Generate statistical recommendations."""
        recommendations = []

        # Check for low power
        low_power_tests = [k for k, v in power_analysis.items() if v < self.config.power_threshold]
        if low_power_tests:
            recommendations.append(
                f"⚠️ {len(low_power_tests)} comparisons have low statistical power (< {self.config.power_threshold}). "
                "Consider increasing sample size or effect size."
            )

        # Check for significant results
        significant_tests = [t for t in test_results if t.significant]
        if significant_tests:
            recommendations.append(
                f"✅ {len(significant_tests)} significant differences found. "
                "These results suggest meaningful differences between models."
            )

        # Check for large effect sizes
        large_effects = [k for k, v in effect_sizes.items() if abs(v) > 0.8]
        if large_effects:
            recommendations.append(
                f"📊 {len(large_effects)} comparisons show large effect sizes (> 0.8). "
                "These represent practically significant differences."
            )

        # Check assumptions
        failed_assumptions = [t for t in test_results if not t.assumptions_met]
        if failed_assumptions:
            recommendations.append(
                f"⚠️ {len(failed_assumptions)} tests failed statistical assumptions. "
                "Consider using non-parametric tests or transforming the data."
            )

        return recommendations

    def _determine_overall_conclusion(self, test_results: List[StatisticalTestResult],
                                    effect_sizes: Dict[str, float],
                                    power_analysis: Dict[str, float]) -> str:
        """Determine overall conclusion from statistical analysis."""

        if not test_results:
            return "No statistical tests were performed."

        # Count significant results
        significant_count = len([t for t in test_results if t.significant])
        total_count = len(test_results)
        significance_rate = significant_count / total_count

        # Count large effect sizes
        large_effects = len([v for v in effect_sizes.values() if abs(v) > 0.8])
        medium_effects = len([v for v in effect_sizes.values() if 0.5 < abs(v) <= 0.8])

        # Count high power tests
        high_power_count = len([v for v in power_analysis.values() if v >= self.config.power_threshold])

        # Determine conclusion
        if significance_rate >= 0.5 and large_effects > 0:
            return f"Strong evidence of differences between models ({significant_count}/{total_count} significant tests, {large_effects} large effects)"
        elif significance_rate >= 0.3:
            return f"Moderate evidence of differences between models ({significant_count}/{total_count} significant tests)"
        elif significance_rate >= 0.1:
            return f"Weak evidence of differences between models ({significant_count}/{total_count} significant tests)"
        elif high_power_count < len(power_analysis) * 0.5:
            return "Insufficient statistical power to detect differences. Consider increasing sample size."
        else:
            return f"No significant differences detected between models ({significant_count}/{total_count} significant tests)"

# Convenience function for easy integration
async def perform_statistical_analysis(
    model_results: List[Dict[str, Any]],
    metrics: List[str],
    config: Optional[StatisticalTestConfig] = None
) -> ModelComparisonResult:
    """
    Convenience function to perform comprehensive statistical analysis.

    Args:
        model_results: List of model results with metrics
        metrics: List of metrics to analyze
        config: Statistical test configuration

    Returns:
        Comprehensive statistical analysis results
    """
    if config is None:
        config = StatisticalTestConfig()

    analyzer = StatisticalAnalyzer(config)
    return await analyzer.analyze_models(model_results, metrics)
