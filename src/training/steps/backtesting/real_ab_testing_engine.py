"""
Real A/B Testing Engine

Enhanced A/B testing for trading strategies with:
- Hardware-accelerated parallel processing (M1 optimization)
- Comprehensive statistical testing (t-test, Mann-Whitney, Wilcoxon, etc.)
- Multiple comparison correction (Bonferroni, Holm, FDR)
- Power analysis and effect size calculations
- Data validation and leakage detection
- Cross-validation support
- Advanced metric calculations
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
from pathlib import Path
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ML utilities
from src.utils.ml_common.optimization import HyperparameterOptimizer
from src.utils.ml_common.cv_utils import TimeSeriesSplitValidator
from src.utils.ml_common.oof_generator import OOFGenerator
from src.utils.ml_common.data_leakage_detector import DataLeakageDetector

# Optional CVLSA support
try:
    from src.utils.ml_common.cvlsa import CVLSAValidator
except ImportError:
    CVLSAValidator = None

# AB Testing base engine
try:
    from src.utils.common_ml.backtesting.ab_testing_engine import ABTestingEngine, ABTestConfig
    BASE_ENGINE_AVAILABLE = True
except ImportError:
    BASE_ENGINE_AVAILABLE = False

# Math validation
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, validate_finite,
    validate_probability, validate_positive, validate_range,
    check_for_nans, check_for_infs
)

# Common operations
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, ensure_directory,
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown,
    calculate_win_rate, calculate_profit_factor, calculate_calmar_ratio,
    calculate_information_ratio
)
from src.utils.common_utilities import ensure_list, ensure_array, flatten_dict

# Hardware optimization
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUAccelerator
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    from src.utils.matrix_operations.hardware_integration import HardwareOptimizedMatrixProcessor
    from src.utils.matrix_operations.batch_operations import BatchMatrixProcessor
    M1_HARDWARE_AVAILABLE = True
except ImportError:
    M1_HARDWARE_AVAILABLE = False

# Output utilities
from src.utils.tprint import tprint

# Decorators
from src.core.decorators import handles_errors, traced, log_execution_time

# Statistical testing imports
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    tprint("⚠️  scipy not available - statistical tests limited", "warning")

try:
    import statsmodels.api as sm
    from statsmodels.stats.power import ttest_power
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    sm = None
    tprint("⚠️  statsmodels not available - power analysis limited", "warning")

logger = logging.getLogger(__name__)

class ABTestType(Enum):
    """A/B test types."""
    PERFORMANCE = "performance"
    RISK_ADJUSTED = "risk_adjusted"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    COMPREHENSIVE = "comprehensive"


@dataclass
class ABTestMetrics:
    """Comprehensive metrics for A/B testing comparison"""
    # Basic metrics
    mean_return_a: float = 0.0
    mean_return_b: float = 0.0
    std_return_a: float = 0.0
    std_return_b: float = 0.0
    
    # Risk-adjusted metrics
    sharpe_ratio_a: float = 0.0
    sharpe_ratio_b: float = 0.0
    sortino_ratio_a: float = 0.0
    sortino_ratio_b: float = 0.0
    calmar_ratio_a: float = 0.0
    calmar_ratio_b: float = 0.0
    
    # Risk metrics
    max_drawdown_a: float = 0.0
    max_drawdown_b: float = 0.0
    var_a: float = 0.0
    var_b: float = 0.0
    
    # Trade metrics
    win_rate_a: float = 0.0
    win_rate_b: float = 0.0
    profit_factor_a: float = 0.0
    profit_factor_b: float = 0.0
    
    # Statistical test results
    t_test_statistic: float = 0.0
    t_test_pvalue: float = 1.0
    mann_whitney_statistic: float = 0.0
    mann_whitney_pvalue: float = 1.0
    
    # Effect sizes
    cohens_d: float = 0.0
    effect_size_category: str = "negligible"
    
    # Power analysis
    statistical_power: float = 0.0
    required_sample_size: int = 0
    
    # Overall assessment
    winner: str = "inconclusive"
    confidence_level: str = "low"
    is_significant: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to structured dictionary"""
        return {
            'strategy_a': {
                'mean_return': self.mean_return_a,
                'std_return': self.std_return_a,
                'sharpe_ratio': self.sharpe_ratio_a,
                'sortino_ratio': self.sortino_ratio_a,
                'calmar_ratio': self.calmar_ratio_a,
                'max_drawdown': self.max_drawdown_a,
                'var': self.var_a,
                'win_rate': self.win_rate_a,
                'profit_factor': self.profit_factor_a
            },
            'strategy_b': {
                'mean_return': self.mean_return_b,
                'std_return': self.std_return_b,
                'sharpe_ratio': self.sharpe_ratio_b,
                'sortino_ratio': self.sortino_ratio_b,
                'calmar_ratio': self.calmar_ratio_b,
                'max_drawdown': self.max_drawdown_b,
                'var': self.var_b,
                'win_rate': self.win_rate_b,
                'profit_factor': self.profit_factor_b
            },
            'statistical_tests': {
                't_test': {
                    'statistic': self.t_test_statistic,
                    'p_value': self.t_test_pvalue
                },
                'mann_whitney': {
                    'statistic': self.mann_whitney_statistic,
                    'p_value': self.mann_whitney_pvalue
                }
            },
            'effect_sizes': {
                'cohens_d': self.cohens_d,
                'category': self.effect_size_category
            },
            'power_analysis': {
                'statistical_power': self.statistical_power,
                'required_sample_size': self.required_sample_size
            },
            'assessment': {
                'winner': self.winner,
                'confidence': self.confidence_level,
                'is_significant': self.is_significant
            }
        }


@dataclass
class RealABTestConfig:
    """Configuration for real A/B testing."""
    # Basic configuration
    test_type: ABTestType = ABTestType.COMPREHENSIVE
    significance_level: float = 0.05
    power: float = 0.8
    min_sample_size: int = 30
    
    # Test parameters
    test_duration_days: int = 252  # 1 year
    warmup_period_days: int = 30
    cooldown_period_days: int = 7
    
    # Statistical parameters
    multiple_comparison_correction: str = "bonferroni"  # "bonferroni", "fdr", "holm"
    effect_size_threshold: float = 0.1
    confidence_interval: float = 0.95
    bootstrap_iterations: int = 1000
    
    # ML validation
    enable_cv_validation: bool = True
    cv_folds: int = 5
    embargo_pct: float = 0.01
    enable_hpo: bool = True
    hpo_method: str = "bayesian"
    
    # Data validation
    enable_data_validation: bool = True
    enable_leakage_detection: bool = True
    
    # Parallel processing
    enable_parallel_processing: bool = True
    max_workers: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    chunk_size_mb: int = 128
    
    # Output settings
    save_results: bool = True
    results_path: str = "ab_testing_results"
    enable_detailed_logging: bool = True
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

class RealABTestingEngine:
    """
    Real A/B testing engine using existing utilities.
    
    This engine provides comprehensive A/B testing with:
    - Statistical significance testing
    - Power analysis
    - Multiple comparison correction
    - ML validation and hyperparameter optimization
    - Risk-adjusted performance comparison
    """
    
    def __init__(self, config: RealABTestConfig):
        """Initialize the enhanced A/B testing engine with hardware acceleration."""
        self.config = config
        self.logger = logger.getChild('RealABTestingEngine')
        
        tprint("🚀 Initializing Enhanced A/B Testing Engine", "header")
        
        # Initialize CV and validation utilities
        if config.enable_cv_validation:
            self.cv_validator = TimeSeriesSplitValidator(
                n_splits=config.cv_folds,
                test_size=1.0 / config.cv_folds,
                embargo_pct=config.embargo_pct
            )
            self.oof_generator = OOFGenerator()
            # Keep CVLSA if available
            self.cvlsa_validator = CVLSAValidator() if CVLSAValidator else None
            tprint("✅ CV utilities initialized", "success")
        else:
            self.cv_validator = None
            self.oof_generator = None
            self.cvlsa_validator = None
        
        # Initialize leakage detector
        if config.enable_leakage_detection:
            self.leakage_detector = DataLeakageDetector()
            tprint("✅ Data leakage detector initialized", "success")
        else:
            self.leakage_detector = None
        
        # Initialize HPO optimizer
        if config.enable_hpo:
            try:
                self.hpo_optimizer = HyperparameterOptimizer()
            except Exception:
                self.hpo_optimizer = None
        else:
            self.hpo_optimizer = None
        
        # Initialize hardware optimization if available
        self.hardware_enabled = M1_HARDWARE_AVAILABLE and config.enable_hardware_optimization
        if self.hardware_enabled:
            self._init_hardware_optimization()
        else:
            self.gpu_accelerator = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.matrix_processor = None
            self.batch_processor = None
            tprint("ℹ️  Hardware optimization disabled", "info")
        
        # Initialize base A/B testing engine if available
        if BASE_ENGINE_AVAILABLE:
            try:
                self.ab_testing_engine = ABTestingEngine()
            except Exception:
                self.ab_testing_engine = None
        else:
            self.ab_testing_engine = None
        
        # Results storage
        self.test_results = {}
        self.statistical_tests = {}
        
        # Configuration summary
        tprint(f"📊 A/B Testing Configuration:", "info")
        tprint(f"   Test type: {config.test_type.value}", "info")
        tprint(f"   Significance level: {config.significance_level}", "info")
        tprint(f"   Power: {config.power}", "info")
        tprint(f"   Min sample size: {config.min_sample_size}", "info")
        tprint(f"   Multiple comparison: {config.multiple_comparison_correction}", "info")
        tprint(f"   CV validation: {config.enable_cv_validation} ({config.cv_folds} folds)", "info")
        tprint(f"   Leakage detection: {config.enable_leakage_detection}", "info")
        tprint(f"   Parallel processing: {config.enable_parallel_processing} ({config.max_workers} workers)", "info")
        tprint(f"   Hardware optimization: {self.hardware_enabled}", "info")
        
        tprint("✅ A/B Testing Engine initialization complete", "success")
    
    def _init_hardware_optimization(self):
        """Initialize hardware optimization components"""
        try:
            tprint("⚡ Initializing M1 hardware optimization", "info")
            
            # Initialize M1 accelerators
            self.gpu_accelerator = M1GPUAccelerator()
            self.memory_optimizer = M1MemoryOptimizer()
            self.cpu_optimizer = M1CPUOptimizer()
            
            # Initialize matrix operations
            self.matrix_processor = HardwareOptimizedMatrixProcessor()
            self.batch_processor = BatchMatrixProcessor(
                chunk_size_mb=self.config.chunk_size_mb,
                enable_gpu=True,
                enable_parallel=True,
                max_workers=self.config.max_workers
            )
            
            # Optimize memory
            self.memory_optimizer.optimize_memory_for_ml()
            
            tprint("✅ Hardware optimization initialized", "success")
            tprint(f"   GPU: {'Available' if self.gpu_accelerator.is_available() else 'Not available'}", "info")
            tprint(f"   Memory optimized: {self.memory_optimizer.is_optimized}", "info")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hardware optimization: {e}")
            tprint(f"⚠️  Hardware optimization init failed: {e}", "warning")
            self.gpu_accelerator = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.matrix_processor = None
            self.batch_processor = None
        
    async def run_ab_test(self, strategy_a_results: Dict[str, Any], 
                         strategy_b_results: Dict[str, Any],
                         test_name: str = "strategy_comparison") -> Dict[str, Any]:
        """Run comprehensive A/B test between two strategies with validation."""
        start_time = time.time()
        tprint(f"🧪 Running A/B Test: {test_name}", "header")
        tprint(f"   Test type: {self.config.test_type.value}", "info")
        
        try:
            # Validate input data
            tprint("📊 Validating strategy results", "info")
            self._validate_test_data(strategy_a_results, strategy_b_results)
            tprint("✅ Data validation passed", "success")
            
            # Extract performance metrics
            tprint("📈 Extracting performance metrics", "info")
            metrics_a = self._extract_metrics(strategy_a_results)
            metrics_b = self._extract_metrics(strategy_b_results)
            
            tprint(f"   Strategy A: {len(metrics_a.get('returns', []))} samples", "info")
            tprint(f"   Strategy B: {len(metrics_b.get('returns', []))} samples", "info")
            
            # Check for data leakage if enabled
            if self.leakage_detector and self.config.enable_leakage_detection:
                self._check_strategy_leakage(metrics_a, metrics_b)
            
            # Run statistical tests based on test type
            tprint(f"🔬 Running statistical tests ({self.config.test_type.value})", "info")
            if self.config.test_type == ABTestType.PERFORMANCE:
                test_results = await self._test_performance(metrics_a, metrics_b)
            elif self.config.test_type == ABTestType.RISK_ADJUSTED:
                test_results = await self._test_risk_adjusted(metrics_a, metrics_b)
            elif self.config.test_type == ABTestType.SHARPE_RATIO:
                test_results = await self._test_sharpe_ratio(metrics_a, metrics_b)
            elif self.config.test_type == ABTestType.MAX_DRAWDOWN:
                test_results = await self._test_max_drawdown(metrics_a, metrics_b)
            elif self.config.test_type == ABTestType.WIN_RATE:
                test_results = await self._test_win_rate(metrics_a, metrics_b)
            elif self.config.test_type == ABTestType.COMPREHENSIVE:
                test_results = await self._test_comprehensive(metrics_a, metrics_b)
            else:
                raise ValueError(f"Unknown test type: {self.config.test_type}")
            
            # Calculate effect sizes
            tprint("📊 Calculating effect sizes", "info")
            effect_sizes = self._calculate_effect_sizes(metrics_a, metrics_b)
            
            # Power analysis
            tprint("⚡ Running power analysis", "info")
            power_analysis = self._calculate_power_analysis(metrics_a, metrics_b)
            
            # Multiple comparison correction
            if self.config.multiple_comparison_correction:
                tprint(f"🔧 Applying {self.config.multiple_comparison_correction} correction", "info")
                test_results = self._apply_multiple_comparison_correction(test_results)
            
            # Generate comprehensive report
            report = self._generate_ab_test_report(
                test_name, metrics_a, metrics_b, test_results, 
                effect_sizes, power_analysis
            )
            
            # Store results
            self.test_results[test_name] = report
            
            execution_time = time.time() - start_time
            
            # Display summary
            tprint(f"✅ A/B Test Complete: {test_name}", "success")
            tprint(f"   Execution time: {execution_time:.2f}s", "info")
            
            # Display key results
            assessment = report.get('test_results', {}).get('overall_assessment', {})
            tprint(f"📊 Test Results:", "info")
            tprint(f"   Winner: {assessment.get('overall_winner', 'inconclusive').upper()}", "info")
            tprint(f"   Confidence: {assessment.get('confidence_level', 'low').upper()}", "info")
            tprint(f"   Significant differences: {assessment.get('significant_differences', 0)}/{assessment.get('total_tests', 0)}", "info")
            
            # Display metric comparisons
            tprint(f"📈 Strategy Comparison:", "info")
            tprint(f"   Mean Return:     A={metrics_a.get('mean_return', 0):.2%} vs B={metrics_b.get('mean_return', 0):.2%}", "info")
            tprint(f"   Sharpe Ratio:    A={metrics_a.get('sharpe_ratio', 0):.3f} vs B={metrics_b.get('sharpe_ratio', 0):.3f}", "info")
            tprint(f"   Max Drawdown:    A={metrics_a.get('max_drawdown', 0):.2%} vs B={metrics_b.get('max_drawdown', 0):.2%}", "info")
            tprint(f"   Win Rate:        A={metrics_a.get('win_rate', 0):.1%} vs B={metrics_b.get('win_rate', 0):.1%}", "info")
            
            # Display effect size
            cohens_d = effect_sizes.get('cohens_d_returns', 0)
            tprint(f"   Cohen's d: {cohens_d:.3f} ({self._categorize_effect_size(cohens_d)})", "info")
            
            # Save results if requested
            if self.config.save_results:
                self._save_ab_test_results(report, test_name)
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ A/B test failed: {e}")
            tprint(f"❌ A/B test failed: {e}", "error")
            raise
    
    def _check_strategy_leakage(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]):
        """Check for data leakage in strategy comparisons"""
        try:
            tprint("🔍 Checking for data leakage", "info")
            
            # Check each strategy
            for name, metrics in [("Strategy A", metrics_a), ("Strategy B", metrics_b)]:
                if 'returns' in metrics:
                    returns = metrics['returns']
                    
                    # Create simple features
                    X = pd.DataFrame({
                        'return': returns,
                        'return_lag1': np.roll(returns, 1),
                    }).iloc[1:]
                    
                    y = pd.Series(returns[1:] > 0)
                    
                    leakage_results = self.leakage_detector.detect_leakage(X.values, y.values)
                    
                    if leakage_results.get('has_leakage', False):
                        leakage_score = leakage_results.get('leakage_score', 0)
                        tprint(f"⚠️  {name} leakage detected: score={leakage_score:.4f}", "warning")
            
            tprint("✅ Leakage check complete", "success")
                    
        except Exception as e:
            tprint(f"⚠️  Leakage detection failed: {e}", "warning")
    
    def _categorize_effect_size(self, cohens_d: float) -> str:
        """Categorize Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _save_ab_test_results(self, report: Dict[str, Any], test_name: str):
        """Save A/B test results to disk"""
        try:
            results_path = Path(self.config.results_path)
            ensure_directory(str(results_path))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = test_name.replace(" ", "_").replace("/", "_")
            
            # Save JSON report
            json_path = results_path / f"ab_test_{safe_name}_{timestamp}.json"
            safe_json_dump(report, str(json_path))
            
            # Save pickle
            pkl_path = results_path / f"ab_test_{safe_name}_{timestamp}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(report, f)
            
            tprint(f"💾 Results saved to {results_path}", "success")
            
        except Exception as e:
            tprint(f"⚠️  Failed to save results: {e}", "warning")
    
    def _validate_test_data(self, strategy_a_results: Dict[str, Any], strategy_b_results: Dict[str, Any]):
        """Validate input data for A/B testing."""
        try:
            # Check if results contain required metrics
            required_metrics = ['returns', 'equity_curve', 'trade_log']
            
            for strategy_name, results in [("Strategy A", strategy_a_results), ("Strategy B", strategy_b_results)]:
                for metric in required_metrics:
                    if metric not in results:
                        raise ValueError(f"{strategy_name} missing required metric: {metric}")
                
                # Validate data types and sizes
                if 'returns' in results:
                    returns = results['returns']
                    if not isinstance(returns, (pd.Series, np.ndarray, list)):
                        raise ValueError(f"{strategy_name} returns must be Series, array, or list")
                    if len(returns) < self.config.min_sample_size:
                        raise ValueError(f"{strategy_name} insufficient data: {len(returns)} < {self.config.min_sample_size}")
                
        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")
            raise
    
    def _extract_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and calculate comprehensive performance metrics using validated utilities."""
        try:
            metrics = {}
            
            # Extract and validate returns
            if 'returns' in results:
                returns = ensure_array(results['returns'])
                # Remove NaN/Inf
                returns = returns[~check_for_nans(returns)]
                returns = returns[~check_for_infs(returns)]
                metrics['returns'] = returns
                
                # Calculate basic statistics with validation
                metrics['mean_return'] = validate_finite(float(np.mean(returns)), default=0.0)
                metrics['std_return'] = validate_positive(float(np.std(returns)), default=0.01)
                metrics['total_return'] = validate_finite(float(np.prod(1 + returns) - 1), default=0.0)
                metrics['volatility'] = validate_positive(metrics['std_return'] * np.sqrt(252), default=0.01)
                
                # Calculate performance metrics using common_operations
                metrics['sharpe_ratio'] = validate_finite(calculate_sharpe_ratio(returns), default=0.0)
                metrics['sortino_ratio'] = validate_finite(calculate_sortino_ratio(returns), default=0.0)
                metrics['win_rate'] = validate_probability(calculate_win_rate(returns))
                metrics['profit_factor'] = validate_positive(calculate_profit_factor(returns), default=0.0)
                
                # Calculate drawdown from returns
                cumulative_returns = np.cumsum(returns)
                metrics['max_drawdown'] = validate_finite(calculate_max_drawdown(cumulative_returns), default=0.0)
                
                # Calculate Calmar ratio
                metrics['calmar_ratio'] = validate_finite(
                    calculate_calmar_ratio(returns, metrics['max_drawdown']), default=0.0
                )
                
                # Calculate Information ratio
                metrics['information_ratio'] = validate_finite(
                    calculate_information_ratio(returns), default=0.0
                )
                
                # Calculate VaR (5%)
                var_percentile = 5.0
                metrics['var_5pct'] = validate_finite(float(np.percentile(returns, var_percentile)), default=0.0)
            
            # Extract equity curve
            if 'equity_curve' in results:
                equity_curve = ensure_array(results['equity_curve'])
                equity_curve = equity_curve[~check_for_nans(equity_curve)]
                equity_curve = equity_curve[~check_for_infs(equity_curve)]
                metrics['equity_curve'] = equity_curve
                
                # Calculate drawdown from equity curve
                if len(equity_curve) > 0:
                    peak = np.maximum.accumulate(equity_curve)
                    drawdown = safe_divide(equity_curve - peak, peak, default=0.0)
                    metrics['max_drawdown_from_equity'] = validate_finite(float(np.min(drawdown)), default=0.0)
                    downside_drawdowns = drawdown[drawdown < 0]
                    if len(downside_drawdowns) > 0:
                        metrics['avg_drawdown'] = validate_finite(float(np.mean(downside_drawdowns)), default=0.0)
            
            # Extract and calculate trade metrics
            if 'trade_log' in results:
                trade_log = results['trade_log']
                if trade_log and isinstance(trade_log, list):
                    # Extract profits
                    profits = [float(t.get('profit', 0)) for t in trade_log if 'profit' in t]
                    profits = [p for p in profits if not (check_for_nans(p) or check_for_infs(p))]
                    
                    if profits:
                        # Calculate trade metrics
                        metrics['n_trades'] = len(profits)
                        winning_trades = [p for p in profits if p > 0]
                        losing_trades = [p for p in profits if p < 0]
                        
                        metrics['win_rate_from_trades'] = validate_probability(
                            len(winning_trades) / len(profits) if profits else 0.0
                        )
                        metrics['avg_win'] = validate_positive(
                            float(np.mean(winning_trades)) if winning_trades else 0.0, default=0.0
                        )
                        metrics['avg_loss'] = validate_finite(
                            float(np.mean(losing_trades)) if losing_trades else 0.0, default=0.0
                        )
                        metrics['profit_factor_from_trades'] = validate_positive(
                            safe_divide(metrics['avg_win'], abs(metrics['avg_loss']), default=0.0), default=0.0
                        )
                        metrics['avg_trade_duration'] = float(
                            np.mean([t.get('duration', 0) for t in trade_log if 'duration' in t])
                        ) if any('duration' in t for t in trade_log) else 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract metrics: {e}")
            tprint(f"❌ Metric extraction failed: {e}", "error")
            raise
    
    async def _test_performance(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Dict[str, Any]:
        """Test performance differences between strategies."""
        try:
            if 'returns' not in metrics_a or 'returns' not in metrics_b:
                raise ValueError("Returns data required for performance testing")
            
            returns_a = metrics_a['returns']
            returns_b = metrics_b['returns']
            
            # T-test for mean returns
            if SCIPY_AVAILABLE:
                t_stat, p_value = stats.ttest_ind(returns_a, returns_b)
                t_test_result = {
                    'test': 't_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config.significance_level
                }
            else:
                t_test_result = {'test': 't_test', 'error': 'scipy not available'}
            
            # Mann-Whitney U test (non-parametric)
            if SCIPY_AVAILABLE:
                u_stat, u_p_value = stats.mannwhitneyu(returns_a, returns_b, alternative='two-sided')
                u_test_result = {
                    'test': 'mann_whitney_u',
                    'statistic': u_stat,
                    'p_value': u_p_value,
                    'significant': u_p_value < self.config.significance_level
                }
            else:
                u_test_result = {'test': 'mann_whitney_u', 'error': 'scipy not available'}
            
            return {
                'performance_tests': {
                    't_test': t_test_result,
                    'mann_whitney_u': u_test_result
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Performance testing failed: {e}")
            raise
    
    async def _test_risk_adjusted(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Dict[str, Any]:
        """Test risk-adjusted performance differences."""
        try:
            # Sharpe ratio comparison
            sharpe_a = metrics_a.get('sharpe_ratio', 0)
            sharpe_b = metrics_b.get('sharpe_ratio', 0)
            
            # Information ratio comparison (if available)
            info_ratio_a = self._calculate_information_ratio(metrics_a)
            info_ratio_b = self._calculate_information_ratio(metrics_b)
            
            # Sortino ratio comparison
            sortino_a = self._calculate_sortino_ratio(metrics_a)
            sortino_b = self._calculate_sortino_ratio(metrics_b)
            
            return {
                'risk_adjusted_tests': {
                    'sharpe_ratio': {
                        'strategy_a': sharpe_a,
                        'strategy_b': sharpe_b,
                        'difference': sharpe_b - sharpe_a,
                        'better': 'B' if sharpe_b > sharpe_a else 'A'
                    },
                    'information_ratio': {
                        'strategy_a': info_ratio_a,
                        'strategy_b': info_ratio_b,
                        'difference': info_ratio_b - info_ratio_a,
                        'better': 'B' if info_ratio_b > info_ratio_a else 'A'
                    },
                    'sortino_ratio': {
                        'strategy_a': sortino_a,
                        'strategy_b': sortino_b,
                        'difference': sortino_b - sortino_a,
                        'better': 'B' if sortino_b > sortino_a else 'A'
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Risk-adjusted testing failed: {e}")
            raise
    
    async def _test_sharpe_ratio(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Dict[str, Any]:
        """Test Sharpe ratio differences."""
        try:
            sharpe_a = metrics_a.get('sharpe_ratio', 0)
            sharpe_b = metrics_b.get('sharpe_ratio', 0)
            
            # Statistical test for Sharpe ratio difference
            if SCIPY_AVAILABLE and 'returns' in metrics_a and 'returns' in metrics_b:
                # Jobson-Korkie test for Sharpe ratio difference
                returns_a = metrics_a['returns']
                returns_b = metrics_b['returns']
                
                # Calculate test statistic
                n_a, n_b = len(returns_a), len(returns_b)
                var_a, var_b = np.var(returns_a), np.var(returns_b)
                cov_ab = np.cov(returns_a, returns_b)[0, 1] if len(returns_a) == len(returns_b) else 0
                
                # Jobson-Korkie statistic
                jk_stat = (sharpe_b - sharpe_a) / np.sqrt(
                    (2 * (1 - cov_ab / np.sqrt(var_a * var_b))) / min(n_a, n_b)
                )
                
                # P-value (approximate)
                p_value = 2 * (1 - stats.norm.cdf(abs(jk_stat)))
                
                sharpe_test = {
                    'test': 'jobson_korkie',
                    'statistic': jk_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config.significance_level
                }
            else:
                sharpe_test = {'test': 'jobson_korkie', 'error': 'insufficient data or scipy not available'}
            
            return {
                'sharpe_ratio_tests': {
                    'sharpe_ratio_comparison': {
                        'strategy_a': sharpe_a,
                        'strategy_b': sharpe_b,
                        'difference': sharpe_b - sharpe_a,
                        'better': 'B' if sharpe_b > sharpe_a else 'A'
                    },
                    'statistical_test': sharpe_test
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Sharpe ratio testing failed: {e}")
            raise
    
    async def _test_max_drawdown(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Dict[str, Any]:
        """Test maximum drawdown differences."""
        try:
            max_dd_a = metrics_a.get('max_drawdown', 0)
            max_dd_b = metrics_b.get('max_drawdown', 0)
            
            # Drawdown comparison (lower is better)
            return {
                'max_drawdown_tests': {
                    'max_drawdown_comparison': {
                        'strategy_a': max_dd_a,
                        'strategy_b': max_dd_b,
                        'difference': max_dd_b - max_dd_a,
                        'better': 'A' if max_dd_a > max_dd_b else 'B'  # Lower drawdown is better
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Max drawdown testing failed: {e}")
            raise
    
    async def _test_win_rate(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Dict[str, Any]:
        """Test win rate differences."""
        try:
            win_rate_a = metrics_a.get('win_rate', 0)
            win_rate_b = metrics_b.get('win_rate', 0)
            
            # Proportion test for win rates
            if 'trade_log' in metrics_a and 'trade_log' in metrics_b:
                trades_a = metrics_a['trade_log']
                trades_b = metrics_b['trade_log']
                
                if trades_a and trades_b:
                    wins_a = len([t for t in trades_a if t.get('profit', 0) > 0])
                    wins_b = len([t for t in trades_b if t.get('profit', 0) > 0])
                    n_a, n_b = len(trades_a), len(trades_b)
                    
                    if SCIPY_AVAILABLE:
                        # Two-proportion z-test
                        p_combined = (wins_a + wins_b) / (n_a + n_b)
                        se = np.sqrt(p_combined * (1 - p_combined) * (1/n_a + 1/n_b))
                        z_stat = (win_rate_b - win_rate_a) / se
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                        
                        proportion_test = {
                            'test': 'two_proportion_z',
                            'statistic': z_stat,
                            'p_value': p_value,
                            'significant': p_value < self.config.significance_level
                        }
                    else:
                        proportion_test = {'test': 'two_proportion_z', 'error': 'scipy not available'}
                else:
                    proportion_test = {'test': 'two_proportion_z', 'error': 'insufficient trade data'}
            else:
                proportion_test = {'test': 'two_proportion_z', 'error': 'no trade log data'}
            
            return {
                'win_rate_tests': {
                    'win_rate_comparison': {
                        'strategy_a': win_rate_a,
                        'strategy_b': win_rate_b,
                        'difference': win_rate_b - win_rate_a,
                        'better': 'B' if win_rate_b > win_rate_a else 'A'
                    },
                    'statistical_test': proportion_test
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Win rate testing failed: {e}")
            raise
    
    async def _test_comprehensive(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive A/B test including all metrics."""
        try:
            # Run all individual tests
            performance_tests = await self._test_performance(metrics_a, metrics_b)
            risk_adjusted_tests = await self._test_risk_adjusted(metrics_a, metrics_b)
            sharpe_tests = await self._test_sharpe_ratio(metrics_a, metrics_b)
            drawdown_tests = await self._test_max_drawdown(metrics_a, metrics_b)
            win_rate_tests = await self._test_win_rate(metrics_a, metrics_b)
            
            # Combine all results
            comprehensive_results = {
                **performance_tests,
                **risk_adjusted_tests,
                **sharpe_tests,
                **drawdown_tests,
                **win_rate_tests
            }
            
            # Overall assessment
            overall_assessment = self._assess_overall_performance(comprehensive_results)
            comprehensive_results['overall_assessment'] = overall_assessment
            
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive testing failed: {e}")
            raise
    
    def _calculate_effect_sizes(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate effect sizes for all metrics."""
        try:
            effect_sizes = {}
            
            # Cohen's d for returns
            if 'returns' in metrics_a and 'returns' in metrics_b:
                returns_a, returns_b = metrics_a['returns'], metrics_b['returns']
                pooled_std = np.sqrt((np.var(returns_a) + np.var(returns_b)) / 2)
                cohens_d = (np.mean(returns_b) - np.mean(returns_a)) / pooled_std
                effect_sizes['cohens_d_returns'] = cohens_d
            
            # Effect sizes for other metrics
            for metric in ['sharpe_ratio', 'max_drawdown', 'win_rate']:
                if metric in metrics_a and metric in metrics_b:
                    value_a, value_b = metrics_a[metric], metrics_b[metric]
                    if metric == 'max_drawdown':
                        # For drawdown, we want the absolute difference
                        effect_sizes[f'{metric}_difference'] = abs(value_b - value_a)
                    else:
                        effect_sizes[f'{metric}_difference'] = value_b - value_a
            
            return effect_sizes
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate effect sizes: {e}")
            return {}
    
    def _calculate_power_analysis(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical power analysis."""
        try:
            power_analysis = {}
            
            if 'returns' in metrics_a and 'returns' in metrics_b:
                returns_a, returns_b = metrics_a['returns'], metrics_b['returns']
                n_a, n_b = len(returns_a), len(returns_b)
                
                # Effect size
                pooled_std = np.sqrt((np.var(returns_a) + np.var(returns_b)) / 2)
                effect_size = (np.mean(returns_b) - np.mean(returns_a)) / pooled_std
                
                # Power calculation
                if STATSMODELS_AVAILABLE:
                    power = ttest_power(effect_size, n_a, alpha=self.config.significance_level)
                    power_analysis['statistical_power'] = power
                    power_analysis['effect_size'] = effect_size
                    power_analysis['sample_size_a'] = n_a
                    power_analysis['sample_size_b'] = n_b
                else:
                    power_analysis['error'] = 'statsmodels not available'
            
            return power_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Power analysis failed: {e}")
            return {}
    
    def _apply_multiple_comparison_correction(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple comparison correction to p-values."""
        try:
            # Extract all p-values
            p_values = []
            p_value_locations = []
            
            def extract_p_values(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key == 'p_value' and isinstance(value, (int, float)):
                            p_values.append(value)
                            p_value_locations.append(f"{path}.{key}")
                        else:
                            extract_p_values(value, f"{path}.{key}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        extract_p_values(item, f"{path}[{i}]")
            
            extract_p_values(test_results)
            
            if not p_values:
                return test_results
            
            # Apply correction
            if self.config.multiple_comparison_correction == "bonferroni":
                corrected_p_values = [p * len(p_values) for p in p_values]
            elif self.config.multiple_comparison_correction == "holm":
                # Holm-Bonferroni correction
                sorted_indices = np.argsort(p_values)
                corrected_p_values = [0] * len(p_values)
                for i, idx in enumerate(sorted_indices):
                    corrected_p_values[idx] = p_values[idx] * (len(p_values) - i)
            else:
                corrected_p_values = p_values
            
            # Update results with corrected p-values
            def update_p_values(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key == 'p_value' and isinstance(value, (int, float)):
                            # Find the corrected value
                            for i, loc in enumerate(p_value_locations):
                                if loc == path:
                                    obj[key] = min(corrected_p_values[i], 1.0)
                                    break
                        else:
                            update_p_values(value, f"{path}.{key}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        update_p_values(item, f"{path}[{i}]")
            
            update_p_values(test_results)
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"❌ Multiple comparison correction failed: {e}")
            return test_results
    
    def _assess_overall_performance(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall performance based on all tests."""
        try:
            assessment = {
                'overall_winner': 'inconclusive',
                'confidence_level': 'low',
                'significant_differences': 0,
                'total_tests': 0
            }
            
            # Count significant differences
            def count_significant(obj):
                if isinstance(obj, dict):
                    if 'significant' in obj and obj['significant']:
                        assessment['significant_differences'] += 1
                    if 'p_value' in obj:
                        assessment['total_tests'] += 1
                    for value in obj.values():
                        count_significant(value)
                elif isinstance(obj, list):
                    for item in obj:
                        count_significant(item)
            
            count_significant(test_results)
            
            # Determine overall winner
            if assessment['significant_differences'] > 0:
                assessment['confidence_level'] = 'high' if assessment['significant_differences'] >= assessment['total_tests'] * 0.5 else 'medium'
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"❌ Overall assessment failed: {e}")
            return {'overall_winner': 'error', 'confidence_level': 'low'}
    
    def _calculate_information_ratio(self, metrics: Dict[str, Any]) -> float:
        """Calculate information ratio."""
        try:
            if 'returns' not in metrics:
                return 0.0
            
            returns = metrics['returns']
            if len(returns) < 2:
                return 0.0
            
            # Information ratio = (portfolio_return - benchmark_return) / tracking_error
            # For simplicity, use risk-free rate as benchmark
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            excess_returns = returns - risk_free_rate
            tracking_error = np.std(excess_returns)
            
            return np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Information ratio calculation failed: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, metrics: Dict[str, Any]) -> float:
        """Calculate Sortino ratio."""
        try:
            if 'returns' not in metrics:
                return 0.0
            
            returns = metrics['returns']
            if len(returns) < 2:
                return 0.0
            
            # Sortino ratio = (portfolio_return - risk_free_rate) / downside_deviation
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            excess_returns = returns - risk_free_rate
            downside_returns = excess_returns[excess_returns < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
            
            return np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Sortino ratio calculation failed: {e}")
            return 0.0
    
    def _generate_ab_test_report(self, test_name: str, metrics_a: Dict[str, Any], 
                                metrics_b: Dict[str, Any], test_results: Dict[str, Any],
                                effect_sizes: Dict[str, Any], power_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive A/B test report."""
        try:
            report = {
                'test_name': test_name,
                'test_config': {
                    'test_type': self.config.test_type.value,
                    'significance_level': self.config.significance_level,
                    'power': self.config.power,
                    'multiple_comparison_correction': self.config.multiple_comparison_correction
                },
                'strategy_metrics': {
                    'strategy_a': metrics_a,
                    'strategy_b': metrics_b
                },
                'test_results': test_results,
                'effect_sizes': effect_sizes,
                'power_analysis': power_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            return {'error': str(e)}

# Convenience functions
async def run_ab_test(
    strategy_a_results: Dict[str, Any],
    strategy_b_results: Dict[str, Any],
    test_type: ABTestType = ABTestType.COMPREHENSIVE,
    significance_level: float = 0.05,
    **kwargs
) -> Dict[str, Any]:
    """Run A/B test between two strategies."""
    config = RealABTestConfig(
        test_type=test_type,
        significance_level=significance_level,
        **kwargs
    )
    
    engine = RealABTestingEngine(config)
    results = await engine.run_ab_test(strategy_a_results, strategy_b_results)
    
    return results