"""
A/B Testing Engine with Statistical Validation

This module provides a comprehensive A/B testing framework with statistical validation,
utilizing M1 optimizations for performance and memory efficiency.
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
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu
import warnings

# M1 Optimization imports
from src.utils.m1_gpu_utils import get_m1_gpu_manager
from src.utils.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.m1_cpu_optimizer import get_m1_cpu_optimizer

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


class TestType(Enum):
    """Types of statistical tests."""
    T_TEST = "t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    WILCOXON = "wilcoxon"


class MetricType(Enum):
    """Types of metrics to test."""
    CONTINUOUS = "continuous"
    BINARY = "binary"
    COUNT = "count"
    RATIO = "ratio"


@dataclass
class StatisticalTest:
    """Statistical test configuration."""
    test_type: TestType
    metric_type: MetricType
    alpha: float = 0.05
    power: float = 0.8
    effect_size: Optional[float] = None
    alternative: str = "two-sided"  # two-sided, greater, less
    equal_var: bool = True  # For t-test
    correction: str = "bonferroni"  # For multiple comparisons


@dataclass
class ABTestConfig:
    """Configuration for A/B testing engine."""
    # Basic configuration
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    
    # Test configuration
    test_name: str
    test_description: str
    control_group_name: str = "control"
    treatment_group_name: str = "treatment"
    
    # Sample configuration
    min_sample_size: int = 100
    max_sample_size: int = 10000
    target_sample_size: Optional[int] = None
    sampling_method: str = "random"  # random, stratified, systematic
    
    # Statistical configuration
    statistical_tests: List[StatisticalTest] = field(default_factory=lambda: [
        StatisticalTest(TestType.T_TEST, MetricType.CONTINUOUS),
        StatisticalTest(TestType.MANN_WHITNEY_U, MetricType.CONTINUOUS)
    ])
    confidence_level: float = 0.95
    multiple_comparison_correction: str = "bonferroni"
    
    # M1 optimization settings
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None
    
    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_profiling: bool = False
    
    # Output settings
    save_detailed_results: bool = True
    generate_plots: bool = True
    output_format: str = "parquet"  # parquet, csv, json
    
    # Validation settings
    min_effect_size: float = 0.01
    max_p_value: float = 0.05
    require_power_analysis: bool = True


@dataclass
class ABTestResults:
    """Results from A/B testing."""
    # Basic info
    test_name: str
    test_description: str
    symbol: str
    exchange: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Sample information
    control_group_size: int
    treatment_group_size: int
    total_sample_size: int
    sampling_method: str
    
    # Test results
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Summary statistics
    control_group_stats: Dict[str, float] = field(default_factory=dict)
    treatment_group_stats: Dict[str, float] = field(default_factory=dict)
    
    # Effect size and power
    effect_size: float = 0.0
    statistical_power: float = 0.0
    minimum_detectable_effect: float = 0.0
    
    # Overall conclusion
    significant_tests: int = 0
    total_tests: int = 0
    overall_conclusion: str = ""
    recommendation: str = ""
    
    # Detailed data
    control_group_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    treatment_group_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Metadata
    config: ABTestConfig = field(default_factory=ABTestConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_used: List[str] = field(default_factory=list)


class ABTestingEngine:
    """A/B testing engine with statistical validation."""
    
    def __init__(self, config: ABTestConfig):
        """Initialize A/B testing engine."""
        self.config = config
        self.logger = logger.getChild('ABTestingEngine')
        
        # Initialize M1 optimizers
        self.m1_gpu = get_m1_gpu_manager() if config.enable_gpu_acceleration else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=config.memory_limit_gb
        ) if config.enable_memory_optimization else None
        self.m1_cpu = get_m1_cpu_optimizer(
            max_workers=config.max_workers
        ) if config.enable_parallel_processing else None
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        
        self.logger.info(f"🚀 ABTestingEngine initialized for {config.test_name}")
        self.logger.info(f"⚡ GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"🧠 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"🔄 Parallel processing: {config.enable_parallel_processing}")
        self.logger.info(f"📊 Statistical tests: {len(config.statistical_tests)}")
    
    @traced(span_name='ab_testing')
    @log_execution_time
    async def execute(
        self, 
        control_data: pd.DataFrame,
        treatment_data: pd.DataFrame,
        metric_columns: List[str],
        **kwargs
    ) -> ABTestResults:
        """Execute A/B testing with statistical validation."""
        
        self.logger.info("🚀 Starting A/B testing...")
        start_time = time.time()
        
        # Validate inputs
        self._validate_inputs(control_data, treatment_data, metric_columns)
        
        # Memory optimization context
        if self.m1_memory:
            with self.m1_memory.optimization_context():
                results = await self._execute_ab_testing(control_data, treatment_data, metric_columns, **kwargs)
        else:
            results = await self._execute_ab_testing(control_data, treatment_data, metric_columns, **kwargs)
        
        execution_time = time.time() - start_time
        results.execution_time = execution_time
        
        # Log memory usage
        if self.m1_memory:
            results.memory_usage_mb = self.m1_memory.get_current_memory_usage_mb()
        
        self.logger.info(f"✅ A/B testing completed in {execution_time:.2f}s")
        self.logger.info(f"📊 Total tests: {results.total_tests}")
        self.logger.info(f"✅ Significant tests: {results.significant_tests}")
        self.logger.info(f"🎯 Overall conclusion: {results.overall_conclusion}")
        
        return results
    
    def _validate_inputs(
        self, 
        control_data: pd.DataFrame, 
        treatment_data: pd.DataFrame, 
        metric_columns: List[str]
    ) -> None:
        """Validate input data and parameters."""
        
        self.logger.info("🔍 Validating A/B test inputs...")
        
        if control_data.empty:
            self.logger.error("❌ Control group data is empty")
            raise ValidationError("Control group data is empty")
        
        if treatment_data.empty:
            self.logger.error("❌ Treatment group data is empty")
            raise ValidationError("Treatment group data is empty")
        
        if not metric_columns:
            self.logger.error("❌ No metric columns specified")
            raise ValidationError("No metric columns specified")
        
        self.logger.info(f"📊 Control group size: {len(control_data):,}")
        self.logger.info(f"📊 Treatment group size: {len(treatment_data):,}")
        self.logger.info(f"📊 Metric columns: {', '.join(metric_columns)}")
        
        # Check if metric columns exist in both datasets
        missing_control = [col for col in metric_columns if col not in control_data.columns]
        missing_treatment = [col for col in metric_columns if col not in treatment_data.columns]
        
        if missing_control:
            self.logger.error(f"❌ Missing columns in control data: {missing_control}")
            raise ValidationError(f"Missing columns in control data: {missing_control}")
        
        if missing_treatment:
            self.logger.error(f"❌ Missing columns in treatment data: {missing_treatment}")
            raise ValidationError(f"Missing columns in treatment data: {missing_treatment}")
        
        # Check sample sizes
        if len(control_data) < self.config.min_sample_size:
            self.logger.error(f"❌ Control group too small: {len(control_data)} < {self.config.min_sample_size}")
            raise ValidationError(f"Control group too small: {len(control_data)} < {self.config.min_sample_size}")
        
        if len(treatment_data) < self.config.min_sample_size:
            self.logger.error(f"❌ Treatment group too small: {len(treatment_data)} < {self.config.min_sample_size}")
            raise ValidationError(f"Treatment group too small: {len(treatment_data)} < {self.config.min_sample_size}")
        
        # Check for sufficient data
        self.logger.info("🔍 Checking data quality for each metric...")
        for col in metric_columns:
            control_valid = control_data[col].notna().sum()
            treatment_valid = treatment_data[col].notna().sum()
            
            self.logger.info(f"   • {col}: Control={control_valid:,}, Treatment={treatment_valid:,}")
            
            if control_valid < 10:
                self.logger.error(f"❌ Insufficient valid data in control group for {col}: {control_valid}")
                raise ValidationError(f"Insufficient valid data in control group for {col}: {control_valid}")
            
            if treatment_valid < 10:
                self.logger.error(f"❌ Insufficient valid data in treatment group for {col}: {treatment_valid}")
                raise ValidationError(f"Insufficient valid data in treatment group for {col}: {treatment_valid}")
        
        self.logger.info("✅ Input validation completed successfully")
    
    async def _execute_ab_testing(
        self, 
        control_data: pd.DataFrame,
        treatment_data: pd.DataFrame,
        metric_columns: List[str],
        **kwargs
    ) -> ABTestResults:
        """Execute the actual A/B testing logic."""
        
        # Sample data if needed
        control_sampled, treatment_sampled = self._sample_data(control_data, treatment_data)
        
        # Calculate summary statistics
        control_stats = self._calculate_summary_stats(control_sampled, metric_columns)
        treatment_stats = self._calculate_summary_stats(treatment_sampled, metric_columns)
        
        # Execute statistical tests
        test_results = []
        for test_config in self.config.statistical_tests:
            for metric in metric_columns:
                result = await self._execute_statistical_test(
                    control_sampled[metric], 
                    treatment_sampled[metric], 
                    test_config, 
                    metric
                )
                test_results.append(result)
        
        # Calculate effect size and power
        effect_size = self._calculate_effect_size(control_sampled, treatment_sampled, metric_columns)
        statistical_power = self._calculate_statistical_power(control_sampled, treatment_sampled, metric_columns)
        
        # Determine overall conclusion
        significant_tests = len([r for r in test_results if r['p_value'] < self.config.max_p_value])
        total_tests = len(test_results)
        overall_conclusion = self._determine_overall_conclusion(significant_tests, total_tests, effect_size)
        recommendation = self._generate_recommendation(overall_conclusion, effect_size, statistical_power)
        
        # Create results
        results = ABTestResults(
            test_name=self.config.test_name,
            test_description=self.config.test_description,
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            timeframe=self.config.timeframe,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,  # Will be set by caller
            control_group_size=len(control_sampled),
            treatment_group_size=len(treatment_sampled),
            total_sample_size=len(control_sampled) + len(treatment_sampled),
            sampling_method=self.config.sampling_method,
            test_results=test_results,
            control_group_stats=control_stats,
            treatment_group_stats=treatment_stats,
            effect_size=effect_size,
            statistical_power=statistical_power,
            minimum_detectable_effect=self._calculate_minimum_detectable_effect(control_sampled, treatment_sampled),
            significant_tests=significant_tests,
            total_tests=total_tests,
            overall_conclusion=overall_conclusion,
            recommendation=recommendation,
            control_group_data=control_sampled,
            treatment_group_data=treatment_sampled,
            config=self.config,
            optimization_used=self._get_optimization_used()
        )
        
        return results
    
    def _sample_data(
        self, 
        control_data: pd.DataFrame, 
        treatment_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Sample data according to configuration."""
        
        # Determine target sample size
        if self.config.target_sample_size:
            target_size = self.config.target_sample_size
        else:
            # Use the smaller of the two groups, but ensure minimum size
            target_size = min(len(control_data), len(treatment_data))
            target_size = max(target_size, self.config.min_sample_size)
            target_size = min(target_size, self.config.max_sample_size)
        
        # Sample control group
        if len(control_data) > target_size:
            if self.config.sampling_method == "random":
                control_sampled = control_data.sample(n=target_size, random_state=42)
            elif self.config.sampling_method == "stratified":
                # Simple stratified sampling (can be enhanced)
                control_sampled = control_data.sample(n=target_size, random_state=42)
            else:
                control_sampled = control_data.sample(n=target_size, random_state=42)
        else:
            control_sampled = control_data.copy()
        
        # Sample treatment group
        if len(treatment_data) > target_size:
            if self.config.sampling_method == "random":
                treatment_sampled = treatment_data.sample(n=target_size, random_state=42)
            elif self.config.sampling_method == "stratified":
                # Simple stratified sampling (can be enhanced)
                treatment_sampled = treatment_data.sample(n=target_size, random_state=42)
            else:
                treatment_sampled = treatment_data.sample(n=target_size, random_state=42)
        else:
            treatment_sampled = treatment_data.copy()
        
        self.logger.info(f"📊 Sampled {len(control_sampled)} control and {len(treatment_sampled)} treatment samples")
        
        return control_sampled, treatment_sampled
    
    def _calculate_summary_stats(self, data: pd.DataFrame, metric_columns: List[str]) -> Dict[str, float]:
        """Calculate summary statistics for the data."""
        stats_dict = {}
        
        for col in metric_columns:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                stats_dict[f"{col}_mean"] = col_data.mean()
                stats_dict[f"{col}_std"] = col_data.std()
                stats_dict[f"{col}_median"] = col_data.median()
                stats_dict[f"{col}_min"] = col_data.min()
                stats_dict[f"{col}_max"] = col_data.max()
                stats_dict[f"{col}_count"] = len(col_data)
                stats_dict[f"{col}_skewness"] = col_data.skew()
                stats_dict[f"{col}_kurtosis"] = col_data.kurtosis()
        
        return stats_dict
    
    async def _execute_statistical_test(
        self, 
        control_data: pd.Series, 
        treatment_data: pd.Series, 
        test_config: StatisticalTest, 
        metric_name: str
    ) -> Dict[str, Any]:
        """Execute a specific statistical test."""
        
        # Clean data
        control_clean = control_data.dropna()
        treatment_clean = treatment_data.dropna()
        
        if len(control_clean) < 2 or len(treatment_clean) < 2:
            return {
                'test_type': test_config.test_type.value,
                'metric': metric_name,
                'p_value': 1.0,
                'statistic': 0.0,
                'significant': False,
                'effect_size': 0.0,
                'confidence_interval': (0.0, 0.0),
                'error': 'Insufficient data'
            }
        
        try:
            if test_config.test_type == TestType.T_TEST:
                result = self._t_test(control_clean, treatment_clean, test_config)
            elif test_config.test_type == TestType.MANN_WHITNEY_U:
                result = self._mann_whitney_u_test(control_clean, treatment_clean, test_config)
            elif test_config.test_type == TestType.CHI_SQUARE:
                result = self._chi_square_test(control_clean, treatment_clean, test_config)
            elif test_config.test_type == TestType.KOLMOGOROV_SMIRNOV:
                result = self._kolmogorov_smirnov_test(control_clean, treatment_clean, test_config)
            else:
                raise ValueError(f"Unsupported test type: {test_config.test_type}")
            
            # Add common fields
            result['test_type'] = test_config.test_type.value
            result['metric'] = metric_name
            result['significant'] = result['p_value'] < test_config.alpha
            result['alpha'] = test_config.alpha
            
            return result
            
        except Exception as e:
            self.logger.error(f"Statistical test failed for {metric_name}: {e}")
            return {
                'test_type': test_config.test_type.value,
                'metric': metric_name,
                'p_value': 1.0,
                'statistic': 0.0,
                'significant': False,
                'effect_size': 0.0,
                'confidence_interval': (0.0, 0.0),
                'error': str(e)
            }
    
    def _t_test(self, control_data: pd.Series, treatment_data: pd.Series, test_config: StatisticalTest) -> Dict[str, Any]:
        """Perform t-test."""
        statistic, p_value = ttest_ind(
            control_data, 
            treatment_data, 
            equal_var=test_config.equal_var,
            alternative=test_config.alternative
        )
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_data) - 1) * control_data.var() + 
                             (len(treatment_data) - 1) * treatment_data.var()) / 
                            (len(control_data) + len(treatment_data) - 2))
        effect_size = (treatment_data.mean() - control_data.mean()) / pooled_std
        
        # Calculate confidence interval for difference in means
        diff = treatment_data.mean() - control_data.mean()
        se_diff = pooled_std * np.sqrt(1/len(control_data) + 1/len(treatment_data))
        alpha = test_config.alpha
        t_critical = stats.t.ppf(1 - alpha/2, len(control_data) + len(treatment_data) - 2)
        ci_lower = diff - t_critical * se_diff
        ci_upper = diff + t_critical * se_diff
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': (ci_lower, ci_upper),
            'control_mean': control_data.mean(),
            'treatment_mean': treatment_data.mean(),
            'control_std': control_data.std(),
            'treatment_std': treatment_data.std()
        }
    
    def _mann_whitney_u_test(self, control_data: pd.Series, treatment_data: pd.Series, test_config: StatisticalTest) -> Dict[str, Any]:
        """Perform Mann-Whitney U test."""
        statistic, p_value = mannwhitneyu(
            control_data, 
            treatment_data, 
            alternative=test_config.alternative
        )
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(control_data), len(treatment_data)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': (0.0, 0.0),  # Not easily calculable for Mann-Whitney U
            'control_median': control_data.median(),
            'treatment_median': treatment_data.median()
        }
    
    def _chi_square_test(self, control_data: pd.Series, treatment_data: pd.Series, test_config: StatisticalTest) -> Dict[str, Any]:
        """Perform chi-square test (for categorical data)."""
        # Convert to categorical if needed
        control_cat = pd.cut(control_data, bins=5, labels=False)
        treatment_cat = pd.cut(treatment_data, bins=5, labels=False)
        
        # Create contingency table
        contingency_table = pd.crosstab(
            pd.concat([pd.Series(control_cat), pd.Series(treatment_cat)]),
            pd.concat([pd.Series(['control'] * len(control_cat)), pd.Series(['treatment'] * len(treatment_cat))])
        )
        
        statistic, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate Cramér's V (effect size)
        n = contingency_table.sum().sum()
        effect_size = np.sqrt(statistic / (n * (min(contingency_table.shape) - 1)))
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': (0.0, 0.0),  # Not easily calculable for chi-square
            'degrees_of_freedom': dof
        }
    
    def _kolmogorov_smirnov_test(self, control_data: pd.Series, treatment_data: pd.Series, test_config: StatisticalTest) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test."""
        statistic, p_value = stats.ks_2samp(control_data, treatment_data)
        
        # Effect size is the KS statistic itself
        effect_size = statistic
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': (0.0, 0.0),  # Not easily calculable for KS test
            'control_ecdf': control_data.quantile([0.25, 0.5, 0.75]).to_dict(),
            'treatment_ecdf': treatment_data.quantile([0.25, 0.5, 0.75]).to_dict()
        }
    
    def _calculate_effect_size(self, control_data: pd.DataFrame, treatment_data: pd.DataFrame, metric_columns: List[str]) -> float:
        """Calculate overall effect size across all metrics."""
        effect_sizes = []
        
        for col in metric_columns:
            control_col = control_data[col].dropna()
            treatment_col = treatment_data[col].dropna()
            
            if len(control_col) > 0 and len(treatment_col) > 0:
                # Calculate Cohen's d
                pooled_std = np.sqrt(((len(control_col) - 1) * control_col.var() + 
                                     (len(treatment_col) - 1) * treatment_col.var()) / 
                                    (len(control_col) + len(treatment_col) - 2))
                
                if pooled_std > 0:
                    effect_size = (treatment_col.mean() - control_col.mean()) / pooled_std
                    effect_sizes.append(abs(effect_size))
        
        return np.mean(effect_sizes) if effect_sizes else 0.0
    
    def _calculate_statistical_power(self, control_data: pd.DataFrame, treatment_data: pd.DataFrame, metric_columns: List[str]) -> float:
        """Calculate statistical power for the test."""
        # Simplified power calculation
        # In practice, this would use more sophisticated methods
        effect_size = self._calculate_effect_size(control_data, treatment_data, metric_columns)
        n = min(len(control_data), len(treatment_data))
        
        # Approximate power calculation
        if effect_size > 0 and n > 0:
            # This is a simplified approximation
            power = min(0.95, 0.5 + 0.3 * effect_size * np.sqrt(n))
        else:
            power = 0.5
        
        return power
    
    def _calculate_minimum_detectable_effect(self, control_data: pd.DataFrame, treatment_data: pd.DataFrame) -> float:
        """Calculate minimum detectable effect size."""
        # Simplified MDE calculation
        n = min(len(control_data), len(treatment_data))
        alpha = self.config.max_p_value
        power = self.config.statistical_tests[0].power if self.config.statistical_tests else 0.8
        
        # Approximate MDE calculation
        if n > 0:
            # This is a simplified approximation
            mde = 2 * np.sqrt(2) * stats.norm.ppf(1 - alpha/2) / np.sqrt(n)
        else:
            mde = 1.0
        
        return mde
    
    def _determine_overall_conclusion(self, significant_tests: int, total_tests: int, effect_size: float) -> str:
        """Determine overall conclusion from test results."""
        
        significance_ratio = significant_tests / total_tests if total_tests > 0 else 0
        
        if significance_ratio >= 0.5 and effect_size >= self.config.min_effect_size:
            return "Significant difference detected"
        elif significance_ratio >= 0.3:
            return "Moderate evidence of difference"
        elif significance_ratio >= 0.1:
            return "Weak evidence of difference"
        else:
            return "No significant difference detected"
    
    def _generate_recommendation(self, conclusion: str, effect_size: float, power: float) -> str:
        """Generate actionable recommendation."""
        
        if "Significant difference" in conclusion and effect_size >= 0.2:
            return "Implement treatment - strong evidence of improvement"
        elif "Significant difference" in conclusion and effect_size >= 0.1:
            return "Consider implementing treatment - moderate improvement"
        elif "Moderate evidence" in conclusion:
            return "Gather more data or refine treatment"
        elif power < 0.8:
            return "Increase sample size for better statistical power"
        else:
            return "No action recommended - insufficient evidence"
    
    def _get_optimization_used(self) -> List[str]:
        """Get list of optimizations used."""
        optimizations = []
        
        if self.config.enable_gpu_acceleration and self.m1_gpu:
            optimizations.append("m1_gpu_acceleration")
        
        if self.config.enable_memory_optimization and self.m1_memory:
            optimizations.append("m1_memory_optimization")
        
        if self.config.enable_parallel_processing and self.m1_cpu:
            optimizations.append("m1_parallel_processing")
        
        return optimizations
    
    async def save_results(self, results: ABTestResults, output_dir: str) -> None:
        """Save A/B test results to disk."""
        ensure_directory(output_dir)
        
        # Save detailed results
        if self.config.save_detailed_results:
            results_file = f"{output_dir}/{self.config.test_name}_ab_test_results.json"
            await safe_json_dump(results_file, results.__dict__)
            self.logger.info(f"💾 Results saved to {results_file}")
        
        # Save test results
        if results.test_results:
            test_results_file = f"{output_dir}/{self.config.test_name}_statistical_tests.parquet"
            test_results_df = pd.DataFrame(results.test_results)
            await self.parquet_utils.save_dataframe(test_results_df, test_results_file)
            self.logger.info(f"💾 Test results saved to {test_results_file}")
        
        # Save control group data
        if not results.control_group_data.empty:
            control_file = f"{output_dir}/{self.config.test_name}_control_group_data.parquet"
            await self.parquet_utils.save_dataframe(results.control_group_data, control_file)
            self.logger.info(f"💾 Control group data saved to {control_file}")
        
        # Save treatment group data
        if not results.treatment_group_data.empty:
            treatment_file = f"{output_dir}/{self.config.test_name}_treatment_group_data.parquet"
            await self.parquet_utils.save_dataframe(results.treatment_group_data, treatment_file)
            self.logger.info(f"💾 Treatment group data saved to {treatment_file}")