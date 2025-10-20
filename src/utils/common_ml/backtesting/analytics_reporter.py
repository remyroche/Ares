"""
Analytics Reporter with Comprehensive Performance Analysis

This module provides comprehensive analytics and reporting capabilities for
backtesting, Monte Carlo simulations, and A/B testing results with M1 optimizations.
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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

# M1 Optimization imports

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
from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)
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

class ReportType(Enum):
    """Types of analytics reports."""
    BACKTESTING = "backtesting"
    MONTE_CARLO = "monte_carlo"
    AB_TESTING = "ab_testing"
    COMPREHENSIVE = "comprehensive"
    PERFORMANCE = "performance"
    RISK = "risk"

class ChartType(Enum):
    """Types of charts to generate."""
    EQUITY_CURVE = "equity_curve"
    DRAWDOWN = "drawdown"
    RETURNS_DISTRIBUTION = "returns_distribution"
    MONTE_CARLO_PATHS = "monte_carlo_paths"
    PERFORMANCE_METRICS = "performance_metrics"
    RISK_METRICS = "risk_metrics"
    AB_TEST_COMPARISON = "ab_test_comparison"
    CORRELATION_MATRIX = "correlation_matrix"

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Basic metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    expected_shortfall: float = 0.0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Advanced metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    jensen_alpha: float = 0.0
    beta: float = 0.0

    # Time-based metrics
    best_month: float = 0.0
    worst_month: float = 0.0
    best_year: float = 0.0
    worst_year: float = 0.0
    positive_months: int = 0
    negative_months: int = 0

    # Consistency metrics
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    recovery_time_days: int = 0

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    # Value at Risk
    var_95: float = 0.0
    var_99: float = 0.0
    var_99_9: float = 0.0

    # Conditional Value at Risk
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    cvar_99_9: float = 0.0

    # Expected Shortfall
    expected_shortfall_95: float = 0.0
    expected_shortfall_99: float = 0.0

    # Drawdown metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    average_drawdown: float = 0.0
    drawdown_frequency: float = 0.0

    # Volatility metrics
    realized_volatility: float = 0.0
    implied_volatility: float = 0.0
    volatility_of_volatility: float = 0.0

    # Tail risk
    tail_ratio: float = 0.0
    tail_expectation: float = 0.0
    extreme_loss_probability: float = 0.0

    # Correlation risk
    correlation_with_market: float = 0.0
    correlation_with_benchmark: float = 0.0

    # Liquidity risk
    liquidity_ratio: float = 0.0
    market_impact: float = 0.0

@dataclass
class AnalyticsConfig:
    """Configuration for analytics reporter."""
    # Basic configuration
    symbol: str
    exchange: str
    timeframe: str
    output_dir: str

    # Report configuration
    report_types: List[ReportType] = field(default_factory=lambda: [ReportType.COMPREHENSIVE])
    chart_types: List[ChartType] = field(default_factory=lambda: [
        ChartType.EQUITY_CURVE,
        ChartType.DRAWDOWN,
        ChartType.RETURNS_DISTRIBUTION,
        ChartType.PERFORMANCE_METRICS
    ])

    # Chart configuration
    chart_style: str = "seaborn-v0_8"
    chart_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    save_format: str = "png"  # png, pdf, svg

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
    save_detailed_reports: bool = True
    generate_html_reports: bool = True
    generate_pdf_reports: bool = False
    include_raw_data: bool = True

    # Analysis settings
    benchmark_symbol: Optional[str] = None
    risk_free_rate: float = 0.02
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])

    # Validation settings
    validate_data_quality: bool = True
    min_data_points: int = 100

class AnalyticsReporter:
    """Comprehensive analytics reporter with M1 optimizations."""

    def __init__(self, config: AnalyticsConfig):
        """Initialize analytics reporter."""
        self.config = config
        self.logger = logger.getChild('AnalyticsReporter')

        # Initialize M1 optimizers
        self.m1_gpu = get_integrated_hardware_manager() if config.enable_gpu_acceleration else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=config.memory_limit_gb
        ) if config.enable_memory_optimization else None
        self.m1_cpu = get_m1_cpu_optimizer(
            max_workers=config.max_workers
        ) if config.enable_parallel_processing else None

        # Initialize utilities
        self.parquet_utils = get_parquet_utils()

        # Set up plotting style
        plt.style.use(self.config.chart_style)
        sns.set_palette("husl")

        # Ensure output directory exists
        ensure_directory(config.output_dir)

        self.logger.info(f"🚀 AnalyticsReporter initialized for {config.symbol}")
        self.logger.info(f"⚡ GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"🧠 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"🔄 Parallel processing: {config.enable_parallel_processing}")
        self.logger.info(f"📊 Report types: {[rt.value for rt in config.report_types]}")

    @traced(span_name='generate_analytics_report')
    async def generate_report(
        self,
        backtesting_results: Optional[Any] = None,
        monte_carlo_results: Optional[Any] = None,
        ab_test_results: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate comprehensive analytics report with M1 optimizations."""

        self.logger.info("🚀 Starting analytics report generation...")
        start_time = time.time()

        # Memory optimization context
        if self.m1_memory:
            with self.m1_memory.optimization_context():
                report = await self._generate_comprehensive_report(
                    backtesting_results, monte_carlo_results, ab_test_results, **kwargs
                )
        else:
            report = await self._generate_comprehensive_report(
                backtesting_results, monte_carlo_results, ab_test_results, **kwargs
            )

        execution_time = time.time() - start_time
        report['execution_time'] = execution_time

        self.logger.info(f"✅ Analytics report generation completed in {execution_time:.2f}s")

        return report

    async def _generate_comprehensive_report(
        self,
        backtesting_results: Optional[Any],
        monte_carlo_results: Optional[Any],
        ab_test_results: Optional[Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""

        report = {
            'symbol': self.config.symbol,
            'exchange': self.config.exchange,
            'timeframe': self.config.timeframe,
            'generated_at': datetime.now().isoformat(),
            'report_types': [rt.value for rt in self.config.report_types],
            'sections': {}
        }

        # Generate performance metrics
        if backtesting_results:
            performance_metrics = self._calculate_performance_metrics(backtesting_results)
            risk_metrics = self._calculate_risk_metrics(backtesting_results)
            report['sections']['performance'] = {
                'metrics': performance_metrics.__dict__,
                'risk_metrics': risk_metrics.__dict__
            }

        # Generate Monte Carlo analysis
        if monte_carlo_results:
            mc_analysis = self._analyze_monte_carlo_results(monte_carlo_results)
            report['sections']['monte_carlo'] = mc_analysis

        # Generate A/B test analysis
        if ab_test_results:
            ab_analysis = self._analyze_ab_test_results(ab_test_results)
            report['sections']['ab_testing'] = ab_analysis

        # Generate charts
        if self.config.chart_types:
            charts = await self._generate_charts(
                backtesting_results, monte_carlo_results, ab_test_results
            )
            report['sections']['charts'] = charts

        # Generate summary
        report['sections']['summary'] = self._generate_summary(report['sections'])

        # Save report
        if self.config.save_detailed_reports:
            await self._save_report(report)

        return report

    def _calculate_performance_metrics(self, backtesting_results: Any) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""

        # Extract basic data
        if hasattr(backtesting_results, 'daily_returns') and not backtesting_results.daily_returns.empty:
            returns = backtesting_results.daily_returns
        elif hasattr(backtesting_results, 'equity_curve') and not backtesting_results.equity_curve.empty:
            equity = backtesting_results.equity_curve
            returns = equity.pct_change().dropna()
        else:
            # Fallback to basic metrics from results
            return PerformanceMetrics(
                total_return=backtesting_results.total_return,
                annualized_return=backtesting_results.annualized_return,
                sharpe_ratio=backtesting_results.sharpe_ratio,
                max_drawdown=backtesting_results.max_drawdown,
                total_trades=backtesting_results.total_trades,
                win_rate=backtesting_results.win_rate
            )

        # Calculate basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0

        # Calculate Sortino ratio
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0

        # Calculate drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Calculate trade metrics
        total_trades = getattr(backtesting_results, 'total_trades', 0)
        winning_trades = getattr(backtesting_results, 'winning_trades', 0)
        losing_trades = getattr(backtesting_results, 'losing_trades', 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate advanced metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Calculate time-based metrics
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()
        positive_months = (monthly_returns > 0).sum()
        negative_months = (monthly_returns < 0).sum()

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            skewness=skewness,
            kurtosis=kurtosis,
            best_month=best_month,
            worst_month=worst_month,
            positive_months=positive_months,
            negative_months=negative_months
        )

    def _calculate_risk_metrics(self, backtesting_results: Any) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""

        # Extract returns data
        if hasattr(backtesting_results, 'daily_returns') and not backtesting_results.daily_returns.empty:
            returns = backtesting_results.daily_returns
        elif hasattr(backtesting_results, 'equity_curve') and not backtesting_results.equity_curve.empty:
            equity = backtesting_results.equity_curve
            returns = equity.pct_change().dropna()
        else:
            return RiskMetrics()

        # Calculate VaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        var_99_9 = np.percentile(returns, 0.1)

        # Calculate CVaR
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        cvar_99_9 = returns[returns <= var_99_9].mean()

        # Calculate drawdown metrics
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max

        max_drawdown = drawdown.min()
        max_drawdown_duration = self._calculate_drawdown_duration(drawdown)
        average_drawdown = drawdown[drawdown < 0].mean()
        drawdown_frequency = (drawdown < 0).sum() / len(drawdown)

        # Calculate volatility metrics
        realized_volatility = returns.std() * np.sqrt(252)
        volatility_of_volatility = returns.rolling(21).std().std() * np.sqrt(252)

        # Calculate tail risk
        tail_ratio = abs(var_95) / abs(var_99) if var_99 != 0 else 0
        tail_expectation = cvar_95
        extreme_loss_probability = (returns < var_99).sum() / len(returns)

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            var_99_9=var_99_9,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            cvar_99_9=cvar_99_9,
            expected_shortfall_95=cvar_95,
            expected_shortfall_99=cvar_99,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            average_drawdown=average_drawdown,
            drawdown_frequency=drawdown_frequency,
            realized_volatility=realized_volatility,
            volatility_of_volatility=volatility_of_volatility,
            tail_ratio=tail_ratio,
            tail_expectation=tail_expectation,
            extreme_loss_probability=extreme_loss_probability
        )

    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0

        if current_period > 0:
            drawdown_periods.append(current_period)

        return max(drawdown_periods) if drawdown_periods else 0

    def _analyze_monte_carlo_results(self, monte_carlo_results: Any) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results."""

        analysis = {
            'simulation_summary': {
                'n_simulations': monte_carlo_results.n_simulations,
                'n_periods': monte_carlo_results.n_periods,
                'simulation_type': monte_carlo_results.simulation_type.value
            },
            'statistics': {
                'mean_final_value': monte_carlo_results.mean_final_value,
                'std_final_value': monte_carlo_results.std_final_value,
                'mean_return': monte_carlo_results.mean_return,
                'std_return': monte_carlo_results.std_return
            },
            'risk_metrics': {
                'var_95': monte_carlo_results.var_95,
                'var_99': monte_carlo_results.var_99,
                'cvar_95': monte_carlo_results.cvar_95,
                'cvar_99': monte_carlo_results.cvar_99
            },
            'percentiles': monte_carlo_results.percentiles,
            'convergence': {
                'achieved': monte_carlo_results.convergence_achieved,
                'iterations': monte_carlo_results.convergence_iterations,
                'error': monte_carlo_results.convergence_error
            }
        }

        return analysis

    def _analyze_ab_test_results(self, ab_test_results: Any) -> Dict[str, Any]:
        """Analyze A/B test results."""

        analysis = {
            'test_summary': {
                'test_name': ab_test_results.test_name,
                'control_group_size': ab_test_results.control_group_size,
                'treatment_group_size': ab_test_results.treatment_group_size,
                'total_sample_size': ab_test_results.total_sample_size
            },
            'statistical_results': {
                'significant_tests': ab_test_results.significant_tests,
                'total_tests': ab_test_results.total_tests,
                'effect_size': ab_test_results.effect_size,
                'statistical_power': ab_test_results.statistical_power
            },
            'conclusion': {
                'overall_conclusion': ab_test_results.overall_conclusion,
                'recommendation': ab_test_results.recommendation
            },
            'group_statistics': {
                'control_group': ab_test_results.control_group_stats,
                'treatment_group': ab_test_results.treatment_group_stats
            }
        }

        return analysis

    async def _generate_charts(
        self,
        backtesting_results: Optional[Any],
        monte_carlo_results: Optional[Any],
        ab_test_results: Optional[Any]
    ) -> Dict[str, str]:
        """Generate charts and save them."""

        charts = {}

        for chart_type in self.config.chart_types:
            try:
                chart_path = await self._generate_chart(
                    chart_type, backtesting_results, monte_carlo_results, ab_test_results
                )
                charts[chart_type.value] = chart_path
            except Exception as e:
                self.logger.error(f"Failed to generate {chart_type.value} chart: {e}")

        return charts

    async def _generate_chart(
        self,
        chart_type: ChartType,
        backtesting_results: Optional[Any],
        monte_carlo_results: Optional[Any],
        ab_test_results: Optional[Any]
    ) -> str:
        """Generate a specific chart."""

        fig, ax = plt.subplots(figsize=self.config.chart_size, dpi=self.config.dpi)

        if chart_type == ChartType.EQUITY_CURVE and backtesting_results:
            await self._plot_equity_curve(ax, backtesting_results)
        elif chart_type == ChartType.DRAWDOWN and backtesting_results:
            await self._plot_drawdown(ax, backtesting_results)
        elif chart_type == ChartType.RETURNS_DISTRIBUTION and backtesting_results:
            await self._plot_returns_distribution(ax, backtesting_results)
        elif chart_type == ChartType.MONTE_CARLO_PATHS and monte_carlo_results:
            await self._plot_monte_carlo_paths(ax, monte_carlo_results)
        elif chart_type == ChartType.PERFORMANCE_METRICS and backtesting_results:
            await self._plot_performance_metrics(ax, backtesting_results)
        elif chart_type == ChartType.RISK_METRICS and backtesting_results:
            await self._plot_risk_metrics(ax, backtesting_results)
        elif chart_type == ChartType.AB_TEST_COMPARISON and ab_test_results:
            await self._plot_ab_test_comparison(ax, ab_test_results)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        # Save chart
        chart_filename = f"{self.config.symbol}_{chart_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.config.save_format}"
        chart_path = f"{self.config.output_dir}/{chart_filename}"

        plt.tight_layout()
        plt.savefig(chart_path, format=self.config.save_format, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        self.logger.info(f"📊 Chart saved: {chart_path}")
        return chart_path

    async def _plot_equity_curve(self, ax, backtesting_results: Any):
        """Plot equity curve."""
        if hasattr(backtesting_results, 'equity_curve') and not backtesting_results.equity_curve.empty:
            equity = backtesting_results.equity_curve
            ax.plot(equity.index, equity.values, linewidth=2, label='Equity Curve')
            ax.set_title(f'Equity Curve - {self.config.symbol}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No equity curve data available', ha='center', va='center', transform=ax.transAxes)

    async def _plot_drawdown(self, ax, backtesting_results: Any):
        """Plot drawdown."""
        if hasattr(backtesting_results, 'daily_returns') and not backtesting_results.daily_returns.empty:
            returns = backtesting_results.daily_returns
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max

            ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red', label='Drawdown')
            ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            ax.set_title(f'Drawdown - {self.config.symbol}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No drawdown data available', ha='center', va='center', transform=ax.transAxes)

    async def _plot_returns_distribution(self, ax, backtesting_results: Any):
        """Plot returns distribution."""
        if hasattr(backtesting_results, 'daily_returns') and not backtesting_results.daily_returns.empty:
            returns = backtesting_results.daily_returns

            ax.hist(returns, bins=50, alpha=0.7, density=True, label='Returns Distribution')
            ax.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
            ax.axvline(returns.median(), color='green', linestyle='--', label=f'Median: {returns.median():.4f}')

            ax.set_title(f'Returns Distribution - {self.config.symbol}')
            ax.set_xlabel('Daily Returns')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No returns data available', ha='center', va='center', transform=ax.transAxes)

    async def _plot_monte_carlo_paths(self, ax, monte_carlo_results: Any):
        """Plot Monte Carlo simulation paths."""
        if hasattr(monte_carlo_results, 'simulated_paths') and monte_carlo_results.simulated_paths.size > 0:
            paths = monte_carlo_results.simulated_paths

            # Plot a subset of paths
            n_paths_to_plot = min(100, paths.shape[0])
            for i in range(0, n_paths_to_plot, 10):
                ax.plot(paths[i], alpha=0.1, color='blue')

            # Plot mean path
            mean_path = np.mean(paths, axis=0)
            ax.plot(mean_path, color='red', linewidth=2, label='Mean Path')

            ax.set_title(f'Monte Carlo Simulation Paths - {self.config.symbol}')
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Monte Carlo paths data available', ha='center', va='center', transform=ax.transAxes)

    async def _plot_performance_metrics(self, ax, backtesting_results: Any):
        """Plot performance metrics comparison."""
        metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        values = [
            getattr(backtesting_results, 'total_return', 0),
            getattr(backtesting_results, 'sharpe_ratio', 0),
            abs(getattr(backtesting_results, 'max_drawdown', 0)),
            getattr(backtesting_results, 'win_rate', 0)
        ]

        bars = ax.bar(metrics, values, alpha=0.7)
        ax.set_title(f'Performance Metrics - {self.config.symbol}')
        ax.set_ylabel('Value')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)

    async def _plot_risk_metrics(self, ax, backtesting_results: Any):
        """Plot risk metrics."""
        risk_metrics = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']
        values = [
            abs(getattr(backtesting_results, 'var_95', 0)),
            abs(getattr(backtesting_results, 'var_99', 0)),
            abs(getattr(backtesting_results, 'cvar_95', 0)),
            abs(getattr(backtesting_results, 'cvar_99', 0))
        ]

        bars = ax.bar(risk_metrics, values, alpha=0.7, color='red')
        ax.set_title(f'Risk Metrics - {self.config.symbol}')
        ax.set_ylabel('Value')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)

    async def _plot_ab_test_comparison(self, ax, ab_test_results: Any):
        """Plot A/B test comparison."""
        if hasattr(ab_test_results, 'control_group_stats') and hasattr(ab_test_results, 'treatment_group_stats'):
            control_stats = ab_test_results.control_group_stats
            treatment_stats = ab_test_results.treatment_group_stats

            # Extract common metrics
            metrics = []
            control_values = []
            treatment_values = []

            for key in control_stats:
                if key.endswith('_mean') and key in treatment_stats:
                    metric_name = key.replace('_mean', '')
                    metrics.append(metric_name)
                    control_values.append(control_stats[key])
                    treatment_values.append(treatment_stats[key])

            if metrics:
                x = np.arange(len(metrics))
                width = 0.35

                ax.bar(x - width/2, control_values, width, label='Control', alpha=0.7)
                ax.bar(x + width/2, treatment_values, width, label='Treatment', alpha=0.7)

                ax.set_title(f'A/B Test Comparison - {ab_test_results.test_name}')
                ax.set_ylabel('Value')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No comparison data available', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No A/B test data available', ha='center', va='center', transform=ax.transAxes)

    def _generate_summary(self, sections: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary."""

        summary = {
            'overall_performance': 'Unknown',
            'risk_level': 'Unknown',
            'recommendation': 'No recommendation available',
            'key_metrics': {},
            'alerts': []
        }

        # Analyze performance section
        if 'performance' in sections:
            perf_metrics = sections['performance']['metrics']

            # Overall performance assessment
            if perf_metrics['sharpe_ratio'] > 1.0:
                summary['overall_performance'] = 'Excellent'
            elif perf_metrics['sharpe_ratio'] > 0.5:
                summary['overall_performance'] = 'Good'
            elif perf_metrics['sharpe_ratio'] > 0.0:
                summary['overall_performance'] = 'Fair'
            else:
                summary['overall_performance'] = 'Poor'

            # Risk level assessment
            if perf_metrics['max_drawdown'] < -0.1:
                summary['risk_level'] = 'High'
            elif perf_metrics['max_drawdown'] < -0.05:
                summary['risk_level'] = 'Medium'
            else:
                summary['risk_level'] = 'Low'

            # Key metrics
            summary['key_metrics'] = {
                'total_return': perf_metrics['total_return'],
                'sharpe_ratio': perf_metrics['sharpe_ratio'],
                'max_drawdown': perf_metrics['max_drawdown'],
                'win_rate': perf_metrics['win_rate']
            }

            # Generate alerts
            if perf_metrics['max_drawdown'] < -0.2:
                summary['alerts'].append('High maximum drawdown detected')
            if perf_metrics['win_rate'] < 0.4:
                summary['alerts'].append('Low win rate detected')
            if perf_metrics['sharpe_ratio'] < 0.0:
                summary['alerts'].append('Negative Sharpe ratio detected')

        # Analyze A/B test section
        if 'ab_testing' in sections:
            ab_results = sections['ab_testing']['conclusion']
            if 'Significant difference' in ab_results['overall_conclusion']:
                summary['recommendation'] = ab_results['recommendation']

        return summary

    async def _save_report(self, report: Dict[str, Any]) -> None:
        """Save comprehensive report to disk."""

        # Save JSON report
        report_file = f"{self.config.output_dir}/{self.config.symbol}_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        await safe_json_dump(report_file, report)
        self.logger.info(f"💾 Analytics report saved to {report_file}")

        # Generate HTML report if requested
        if self.config.generate_html_reports:
            html_file = report_file.replace('.json', '.html')
            await self._generate_html_report(report, html_file)
            self.logger.info(f"💾 HTML report saved to {html_file}")

    async def _generate_html_report(self, report: Dict[str, Any], html_file: str) -> None:
        """Generate HTML report."""

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analytics Report - {self.config.symbol}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
                .alert {{ color: red; font-weight: bold; }}
                .success {{ color: green; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Analytics Report - {self.config.symbol}</h1>
                <p>Generated: {report['generated_at']}</p>
                <p>Exchange: {report['exchange']} | Timeframe: {report['timeframe']}</p>
            </div>
        """

        # Add summary section
        if 'summary' in report['sections']:
            summary = report['sections']['summary']
            html_content += f"""
            <div class="section">
                <h2>Executive Summary</h2>
                <p><strong>Overall Performance:</strong> <span class="{'success' if summary['overall_performance'] in ['Excellent', 'Good'] else 'alert'}">{summary['overall_performance']}</span></p>
                <p><strong>Risk Level:</strong> {summary['risk_level']}</p>
                <p><strong>Recommendation:</strong> {summary['recommendation']}</p>
            </div>
            """

        # Add performance section
        if 'performance' in report['sections']:
            perf = report['sections']['performance']['metrics']
            html_content += f"""
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="metric">Total Return: {perf['total_return']:.2%}</div>
                <div class="metric">Sharpe Ratio: {perf['sharpe_ratio']:.2f}</div>
                <div class="metric">Max Drawdown: {perf['max_drawdown']:.2%}</div>
                <div class="metric">Win Rate: {perf['win_rate']:.2%}</div>
            </div>
            """

        # Add charts section
        if 'charts' in report['sections']:
            html_content += """
            <div class="section">
                <h2>Charts</h2>
            """
            for chart_name, chart_path in report['sections']['charts'].items():
                html_content += f'<p><img src="{Path(chart_path).name}" alt="{chart_name}" style="max-width: 100%;"></p>'
            html_content += "</div>"

        html_content += """
        </body>
        </html>
        """

        with open(html_file, 'w') as f:
            f.write(html_content)
