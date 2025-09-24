"""
Real A/B Testing Engine

This module provides comprehensive A/B testing for trading strategies using
existing utilities from src/utils/ for statistical analysis and ML validation.
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import existing utilities
from src.utils.ml_common.optimization import HyperparameterOptimizer
from src.utils.ml_common.cvlsa import CVLSAValidator
from src.utils.common_ml.backtesting.ab_testing_engine import ABTestingEngine, ABTestConfig
from src.utils.common_operations import safe_json_dump, safe_json_load, ensure_directory
from src.utils.math_validation import safe_divide, safe_log, safe_sqrt, validate_finite
from src.core.decorators import handles_errors, traced, log_execution_time

# Statistical testing imports
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

try:
    import statsmodels.api as sm
    from statsmodels.stats.power import ttest_power
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    sm = None

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
    
    # ML validation
    enable_cv_validation: bool = True
    enable_hpo: bool = True
    hpo_method: str = "bayesian"
    
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
        """Initialize the real A/B testing engine."""
        self.config = config
        self.logger = logger.getChild('RealABTestingEngine')
        
        # Initialize ML utilities
        self.cv_validator = CVLSAValidator() if config.enable_cv_validation else None
        self.hpo_optimizer = HyperparameterOptimizer() if config.enable_hpo else None
        
        # Initialize A/B testing engine
        self.ab_testing_engine = ABTestingEngine()
        
        # Results storage
        self.test_results = {}
        self.statistical_tests = {}
        
    async def run_ab_test(self, strategy_a_results: Dict[str, Any], 
                         strategy_b_results: Dict[str, Any],
                         test_name: str = "strategy_comparison") -> Dict[str, Any]:
        """Run comprehensive A/B test between two strategies."""
        self.logger.info(f"🧪 Running A/B test: {test_name}")
        
        try:
            # Validate input data
            self._validate_test_data(strategy_a_results, strategy_b_results)
            
            # Extract performance metrics
            metrics_a = self._extract_metrics(strategy_a_results)
            metrics_b = self._extract_metrics(strategy_b_results)
            
            # Run statistical tests based on test type
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
            effect_sizes = self._calculate_effect_sizes(metrics_a, metrics_b)
            
            # Power analysis
            power_analysis = self._calculate_power_analysis(metrics_a, metrics_b)
            
            # Multiple comparison correction
            if self.config.multiple_comparison_correction:
                test_results = self._apply_multiple_comparison_correction(test_results)
            
            # Generate comprehensive report
            report = self._generate_ab_test_report(
                test_name, metrics_a, metrics_b, test_results, 
                effect_sizes, power_analysis
            )
            
            # Store results
            self.test_results[test_name] = report
            
            self.logger.info(f"✅ A/B test completed: {test_name}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ A/B test failed: {e}")
            raise
    
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
        """Extract performance metrics from strategy results."""
        try:
            metrics = {}
            
            # Extract returns
            if 'returns' in results:
                returns = results['returns']
                if isinstance(returns, pd.Series):
                    returns = returns.values
                metrics['returns'] = np.array(returns)
            
            # Extract equity curve
            if 'equity_curve' in results:
                equity_curve = results['equity_curve']
                if isinstance(equity_curve, list):
                    equity_curve = np.array(equity_curve)
                metrics['equity_curve'] = equity_curve
            
            # Extract trade log
            if 'trade_log' in results:
                metrics['trade_log'] = results['trade_log']
            
            # Calculate derived metrics
            if 'returns' in metrics:
                returns = metrics['returns']
                metrics['mean_return'] = np.mean(returns)
                metrics['std_return'] = np.std(returns)
                metrics['sharpe_ratio'] = metrics['mean_return'] / metrics['std_return'] if metrics['std_return'] > 0 else 0
                metrics['total_return'] = np.prod(1 + returns) - 1
                metrics['volatility'] = metrics['std_return'] * np.sqrt(252)
            
            # Calculate drawdown metrics
            if 'equity_curve' in metrics:
                equity_curve = metrics['equity_curve']
                peak = np.maximum.accumulate(equity_curve)
                drawdown = (equity_curve - peak) / peak
                metrics['max_drawdown'] = np.min(drawdown)
                metrics['avg_drawdown'] = np.mean(drawdown[drawdown < 0])
            
            # Calculate trade metrics
            if 'trade_log' in metrics:
                trade_log = metrics['trade_log']
                if trade_log:
                    profits = [t.get('profit', 0) for t in trade_log if 'profit' in t]
                    if profits:
                        metrics['win_rate'] = len([p for p in profits if p > 0]) / len(profits)
                        metrics['avg_win'] = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
                        metrics['avg_loss'] = np.mean([p for p in profits if p < 0]) if any(p < 0 for p in profits) else 0
                        metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract metrics: {e}")
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