"""
Unified Backtesting Orchestrator

This module provides a unified orchestrator that coordinates all backtesting
components and provides a single interface for comprehensive backtesting analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import unified backtesting components
from .backtesting_engine import BacktestingEngine, BacktestingConfig, BacktestingResult
from .monte_carlo_engine import MonteCarloEngine, MonteCarloConfig, MonteCarloResult
from .performance_attribution import PerformanceAttribution, PerformanceAttributionConfig, PerformanceMetrics
from .walk_forward_analyzer import WalkForwardAnalyzer, WalkForwardConfig, WalkForwardResult
from .data_manager import BacktestingDataManager, DataManagerConfig
from .risk_analyzer import RiskAnalyzer, RiskAnalysisConfig, RiskMetrics

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for unified backtesting orchestrator."""
    
    # Core backtesting configuration
    backtesting_config: BacktestingConfig = field(default_factory=BacktestingConfig)
    
    # Component configurations
    monte_carlo_config: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    performance_config: PerformanceAttributionConfig = field(default_factory=PerformanceAttributionConfig)
    walk_forward_config: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    data_config: DataManagerConfig = field(default_factory=DataManagerConfig)
    risk_config: RiskAnalysisConfig = field(default_factory=RiskAnalysisConfig)
    
    # Analysis options
    enable_monte_carlo: bool = True
    enable_walk_forward: bool = True
    enable_performance_attribution: bool = True
    enable_risk_analysis: bool = True
    enable_regime_analysis: bool = True
    
    # Output options
    save_all_results: bool = True
    results_directory: str = "backtesting_results"
    generate_reports: bool = True
    enable_plotting: bool = True
    
    # Performance options
    parallel_processing: bool = True
    max_workers: Optional[int] = None


@dataclass
class UnifiedBacktestingResult:
    """Unified result from comprehensive backtesting analysis."""
    
    # Core backtesting result
    backtesting_result: BacktestingResult
    
    # Additional analysis results
    monte_carlo_result: Optional[MonteCarloResult] = None
    walk_forward_result: Optional[WalkForwardResult] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    risk_metrics: Optional[RiskMetrics] = None
    
    # Summary metrics
    overall_score: float = 0.0
    risk_adjusted_score: float = 0.0
    stability_score: float = 0.0
    
    # Metadata
    execution_time: float = 0.0
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    config: OrchestratorConfig = field(default_factory=OrchestratorConfig)


class UnifiedBacktestingOrchestrator:
    """
    Unified backtesting orchestrator that coordinates all backtesting components.
    
    Provides a single interface for comprehensive backtesting analysis including
    historical backtesting, Monte Carlo simulation, walk-forward analysis,
    performance attribution, and risk analysis.
    """
    
    def __init__(self, config: OrchestratorConfig):
        """Initialize the unified orchestrator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Create results directory
        if self.config.save_all_results:
            self.results_dir = Path(self.config.results_directory)
            self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_components(self):
        """Initialize all backtesting components."""
        # Initialize data manager
        self.data_manager = BacktestingDataManager(self.config.data_config)
        
        # Initialize backtesting engine
        self.backtesting_engine = BacktestingEngine(self.config.backtesting_config)
        
        # Initialize Monte Carlo engine
        if self.config.enable_monte_carlo:
            self.monte_carlo_engine = MonteCarloEngine(self.config.monte_carlo_config)
        else:
            self.monte_carlo_engine = None
        
        # Initialize walk-forward analyzer
        if self.config.enable_walk_forward:
            self.walk_forward_analyzer = WalkForwardAnalyzer(self.config.walk_forward_config)
        else:
            self.walk_forward_analyzer = None
        
        # Initialize performance attribution
        if self.config.enable_performance_attribution:
            self.performance_attribution = PerformanceAttribution(self.config.performance_config)
        else:
            self.performance_attribution = None
        
        # Initialize risk analyzer
        if self.config.enable_risk_analysis:
            self.risk_analyzer = RiskAnalyzer(self.config.risk_config)
        else:
            self.risk_analyzer = None
    
    def run_comprehensive_analysis(
        self,
        model: Any,
        data: Optional[pd.DataFrame] = None,
        regime_detector: Optional[Any] = None,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> UnifiedBacktestingResult:
        """
        Run comprehensive backtesting analysis.
        
        Args:
            model: Trading model or strategy to analyze
            data: Market data (optional, will be loaded if not provided)
            regime_detector: Regime detection system (optional)
            benchmark_data: Benchmark data for comparison (optional)
            
        Returns:
            UnifiedBacktestingResult with comprehensive analysis
        """
        start_time = datetime.now()
        self.logger.info("Starting comprehensive backtesting analysis")
        
        try:
            # Load and prepare data
            if data is None:
                data = self.data_manager.load_data()
            else:
                data = self.data_manager._process_data(data)
            
            # Run core backtesting
            self.logger.info("Running core backtesting analysis")
            backtesting_result = self.backtesting_engine.run_backtest(model, data, regime_detector)
            
            # Run Monte Carlo simulation
            monte_carlo_result = None
            if self.config.enable_monte_carlo and self.monte_carlo_engine:
                self.logger.info("Running Monte Carlo simulation")
                monte_carlo_result = self.monte_carlo_engine.run_simulation(model, data, regime_detector)
            
            # Run walk-forward analysis
            walk_forward_result = None
            if self.config.enable_walk_forward and self.walk_forward_analyzer:
                self.logger.info("Running walk-forward analysis")
                walk_forward_result = self.walk_forward_analyzer.analyze(model, data, regime_detector)
            
            # Run performance attribution
            performance_metrics = None
            if (self.config.enable_performance_attribution and 
                self.performance_attribution and 
                backtesting_result.equity_curve is not None):
                
                self.logger.info("Running performance attribution analysis")
                strategy_returns = backtesting_result.equity_curve['equity'].pct_change().dropna()
                benchmark_returns = None
                
                if benchmark_data is not None:
                    if 'returns' in benchmark_data.columns:
                        benchmark_returns = benchmark_data['returns']
                    elif 'close' in benchmark_data.columns:
                        benchmark_returns = benchmark_data['close'].pct_change()
                
                performance_metrics = self.performance_attribution.analyze_performance(
                    strategy_returns, benchmark_returns
                )
            
            # Run risk analysis
            risk_metrics = None
            if (self.config.enable_risk_analysis and 
                self.risk_analyzer and 
                backtesting_result.equity_curve is not None):
                
                self.logger.info("Running risk analysis")
                returns = backtesting_result.equity_curve['equity'].pct_change().dropna()
                market_returns = None
                
                if benchmark_data is not None:
                    if 'returns' in benchmark_data.columns:
                        market_returns = benchmark_data['returns']
                    elif 'close' in benchmark_data.columns:
                        market_returns = benchmark_data['close'].pct_change()
                
                risk_metrics_dict = self.risk_analyzer.analyze(returns, market_returns)
                
                # Convert to RiskMetrics object
                risk_metrics = RiskMetrics(
                    var_95=risk_metrics_dict.get('var_95', 0),
                    var_99=risk_metrics_dict.get('var_99', 0),
                    cvar_95=risk_metrics_dict.get('cvar_95', 0),
                    cvar_99=risk_metrics_dict.get('cvar_99', 0),
                    expected_shortfall=risk_metrics_dict.get('expected_shortfall', 0),
                    tail_ratio=risk_metrics_dict.get('tail_ratio', 0),
                    max_drawdown=risk_metrics_dict.get('max_drawdown', 0),
                    avg_drawdown=risk_metrics_dict.get('avg_drawdown', 0),
                    drawdown_duration=risk_metrics_dict.get('drawdown_duration', 0),
                    recovery_time=risk_metrics_dict.get('recovery_time', 0),
                    realized_volatility=risk_metrics_dict.get('realized_volatility', 0),
                    volatility_of_volatility=risk_metrics_dict.get('volatility_of_volatility', 0),
                    correlation_to_market=risk_metrics_dict.get('correlation_to_market', 0),
                    correlation_stability=risk_metrics_dict.get('correlation_stability', 0),
                    stress_test_results=risk_metrics_dict.get('stress_test_results'),
                    returns_series=returns
                )
            
            # Calculate overall scores
            overall_scores = self._calculate_overall_scores(
                backtesting_result, monte_carlo_result, walk_forward_result,
                performance_metrics, risk_metrics
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create unified result
            unified_result = UnifiedBacktestingResult(
                backtesting_result=backtesting_result,
                monte_carlo_result=monte_carlo_result,
                walk_forward_result=walk_forward_result,
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                overall_score=overall_scores['overall'],
                risk_adjusted_score=overall_scores['risk_adjusted'],
                stability_score=overall_scores['stability'],
                execution_time=execution_time,
                config=self.config
            )
            
            # Save results if requested
            if self.config.save_all_results:
                self._save_results(unified_result)
            
            # Generate reports if requested
            if self.config.generate_reports:
                self._generate_reports(unified_result)
            
            self.logger.info(f"Comprehensive analysis completed in {execution_time:.2f} seconds")
            return unified_result
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            raise
    
    def _calculate_overall_scores(
        self,
        backtesting_result: BacktestingResult,
        monte_carlo_result: Optional[MonteCarloResult],
        walk_forward_result: Optional[WalkForwardResult],
        performance_metrics: Optional[PerformanceMetrics],
        risk_metrics: Optional[RiskMetrics]
    ) -> Dict[str, float]:
        """Calculate overall performance scores."""
        scores = {'overall': 0, 'risk_adjusted': 0, 'stability': 0}
        
        # Overall score based on returns and Sharpe ratio
        return_score = min(backtesting_result.total_return * 2, 1.0)  # Cap at 1.0
        sharpe_score = min(max(backtesting_result.sharpe_ratio / 2.0, 0), 1.0)  # Scale to 0-1
        scores['overall'] = (return_score + sharpe_score) / 2
        
        # Risk-adjusted score
        if risk_metrics:
            drawdown_score = max(0, 1 + risk_metrics.max_drawdown)  # Higher is better
            var_score = max(0, 1 + risk_metrics.var_95)  # Higher is better
            scores['risk_adjusted'] = (drawdown_score + var_score) / 2
        else:
            scores['risk_adjusted'] = scores['overall']
        
        # Stability score
        if walk_forward_result:
            scores['stability'] = (walk_forward_result.performance_stability + 
                                 walk_forward_result.parameter_stability) / 2
        else:
            scores['stability'] = 0.5  # Default moderate stability
        
        return scores
    
    def _save_results(self, result: UnifiedBacktestingResult):
        """Save all results to files."""
        timestamp = result.analysis_timestamp.strftime('%Y%m%d_%H%M%S')
        
        # Save summary
        summary_file = self.results_dir / f"unified_analysis_summary_{timestamp}.json"
        summary_data = {
            'overall_score': result.overall_score,
            'risk_adjusted_score': result.risk_adjusted_score,
            'stability_score': result.stability_score,
            'execution_time': result.execution_time,
            'backtesting_metrics': {
                'total_return': result.backtesting_result.total_return,
                'sharpe_ratio': result.backtesting_result.sharpe_ratio,
                'max_drawdown': result.backtesting_result.max_drawdown,
                'total_trades': result.backtesting_result.total_trades
            }
        }
        
        if result.monte_carlo_result:
            summary_data['monte_carlo_metrics'] = {
                'expected_return': result.monte_carlo_result.mean_return,
                'var_95': result.monte_carlo_result.var_95,
                'probability_of_loss': result.monte_carlo_result.probability_of_loss
            }
        
        if result.walk_forward_result:
            summary_data['walk_forward_metrics'] = {
                'performance_stability': result.walk_forward_result.performance_stability,
                'parameter_stability': result.walk_forward_result.parameter_stability,
                'n_periods': result.walk_forward_result.n_periods
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        # Save detailed data
        if result.backtesting_result.equity_curve is not None:
            equity_file = self.results_dir / f"equity_curve_{timestamp}.parquet"
            result.backtesting_result.equity_curve.to_parquet(equity_file)
        
        if result.walk_forward_result and result.walk_forward_result.equity_curve is not None:
            wf_file = self.results_dir / f"walk_forward_equity_{timestamp}.parquet"
            result.walk_forward_result.equity_curve.to_parquet(wf_file)
        
        self.logger.info(f"Results saved to {self.results_dir}")
    
    def _generate_reports(self, result: UnifiedBacktestingResult):
        """Generate comprehensive reports."""
        timestamp = result.analysis_timestamp.strftime('%Y%m%d_%H%M%S')
        
        # Generate unified report
        report_file = self.results_dir / f"unified_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("UNIFIED BACKTESTING ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall summary
            f.write("OVERALL SUMMARY:\n")
            f.write(f"Overall Score: {result.overall_score:.3f}\n")
            f.write(f"Risk-Adjusted Score: {result.risk_adjusted_score:.3f}\n")
            f.write(f"Stability Score: {result.stability_score:.3f}\n")
            f.write(f"Execution Time: {result.execution_time:.2f} seconds\n\n")
            
            # Backtesting results
            f.write("BACKTESTING RESULTS:\n")
            f.write(f"Total Return: {result.backtesting_result.total_return:.2%}\n")
            f.write(f"Sharpe Ratio: {result.backtesting_result.sharpe_ratio:.3f}\n")
            f.write(f"Max Drawdown: {result.backtesting_result.max_drawdown:.2%}\n")
            f.write(f"Total Trades: {result.backtesting_result.total_trades}\n\n")
            
            # Monte Carlo results
            if result.monte_carlo_result:
                f.write("MONTE CARLO SIMULATION:\n")
                f.write(f"Expected Return: {result.monte_carlo_result.mean_return:.2%}\n")
                f.write(f"VaR (95%): {result.monte_carlo_result.var_95:.2%}\n")
                f.write(f"Probability of Loss: {result.monte_carlo_result.probability_of_loss:.2%}\n\n")
            
            # Walk-forward results
            if result.walk_forward_result:
                f.write("WALK-FORWARD ANALYSIS:\n")
                f.write(f"Performance Stability: {result.walk_forward_result.performance_stability:.3f}\n")
                f.write(f"Parameter Stability: {result.walk_forward_result.parameter_stability:.3f}\n")
                f.write(f"Number of Periods: {result.walk_forward_result.n_periods}\n\n")
            
            # Risk analysis
            if result.risk_metrics:
                f.write("RISK ANALYSIS:\n")
                f.write(f"VaR (95%): {result.risk_metrics.var_95:.2%}\n")
                f.write(f"CVaR (95%): {result.risk_metrics.cvar_95:.2%}\n")
                f.write(f"Max Drawdown: {result.risk_metrics.max_drawdown:.2%}\n")
                f.write(f"Realized Volatility: {result.risk_metrics.realized_volatility:.2%}\n\n")
            
            # Performance attribution
            if result.performance_metrics:
                f.write("PERFORMANCE ATTRIBUTION:\n")
                f.write(f"Excess Return: {result.performance_metrics.excess_return:.2%}\n")
                f.write(f"Tracking Error: {result.performance_metrics.tracking_error:.2%}\n")
                f.write(f"Beta: {result.performance_metrics.beta:.3f}\n")
                f.write(f"Alpha: {result.performance_metrics.alpha:.2%}\n\n")
        
        self.logger.info(f"Reports generated in {self.results_dir}")


# Convenience functions
def run_unified_backtest(
    model: Any,
    data: Optional[pd.DataFrame] = None,
    config: Optional[OrchestratorConfig] = None
) -> UnifiedBacktestingResult:
    """Run unified backtesting with default configuration."""
    if config is None:
        config = OrchestratorConfig()
    
    orchestrator = UnifiedBacktestingOrchestrator(config)
    return orchestrator.run_comprehensive_analysis(model, data)


def create_quick_config() -> OrchestratorConfig:
    """Create a quick configuration for basic backtesting."""
    return OrchestratorConfig(
        enable_monte_carlo=False,
        enable_walk_forward=False,
        enable_performance_attribution=True,
        enable_risk_analysis=True,
        generate_reports=True
    )


def create_full_config() -> OrchestratorConfig:
    """Create a full configuration for comprehensive backtesting."""
    return OrchestratorConfig(
        enable_monte_carlo=True,
        enable_walk_forward=True,
        enable_performance_attribution=True,
        enable_risk_analysis=True,
        generate_reports=True,
        save_all_results=True
    )