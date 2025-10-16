"""
Walk-Forward Analysis for TAS

Comprehensive walk-forward analysis for tree architecture search including
out-of-sample testing, rolling window analysis, and performance validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from .backtesting_engine import BacktestingEngine, BacktestingConfig, BacktestingResult

logger = logging.getLogger(__name__)


class WalkForwardMode(Enum):
    """Walk-forward analysis modes."""
    ROLLING = "rolling"
    EXPANDING = "expanding"
    FIXED = "fixed"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    
    # Walk-forward parameters
    training_window: int = 252  # Trading days
    testing_window: int = 63    # Trading days
    step_size: int = 21         # Trading days
    mode: WalkForwardMode = WalkForwardMode.ROLLING
    
    # Data parameters
    min_training_samples: int = 100
    min_testing_samples: int = 20
    
    # Performance thresholds
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = 0.15
    min_win_rate: float = 0.4
    
    # Analysis parameters
    enable_regime_analysis: bool = True
    enable_performance_attribution: bool = True
    enable_risk_analysis: bool = True
    
    # Output parameters
    save_individual_results: bool = True
    save_summary: bool = True
    results_directory: str = "walk_forward_results"
    
    # Advanced parameters
    enable_parameter_optimization: bool = False
    optimization_metric: str = "sharpe_ratio"  # "sharpe_ratio", "total_return", "calmar_ratio"
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis."""
    
    # Overall metrics
    n_periods: int
    successful_periods: int
    failed_periods: int
    success_rate: float
    
    # Performance metrics
    average_return: float
    average_sharpe: float
    average_drawdown: float
    total_return: float
    cumulative_return: float
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    
    # Period-by-period results
    period_results: List[Dict[str, Any]]
    period_returns: List[float]
    period_sharpe: List[float]
    period_drawdown: List[float]
    
    # Regime analysis
    regime_performance: Dict[str, float]
    regime_stability: Dict[str, float]
    
    # Time series
    equity_curve: pd.Series
    returns_series: pd.Series
    drawdown_series: pd.Series
    
    # Metadata
    analysis_period: Tuple[datetime, datetime]
    execution_time: float
    config: WalkForwardConfig


class WalkForwardAnalyzer:
    """
    Comprehensive walk-forward analyzer for TAS.
    
    Provides rolling window analysis, out-of-sample testing,
    and performance validation across multiple time periods.
    """
    
    def __init__(self, config: WalkForwardConfig):
        """Initialize walk-forward analyzer.
        
        Args:
            config: Walk-forward configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Analysis state
        self.results = None
        self.period_results = []
        self.equity_curve = []
        self.returns_series = []
        
        self.logger.info("✅ Walk-Forward Analyzer initialized")
        self.logger.info(f"📅 Training window: {config.training_window} days")
        self.logger.info(f"📅 Testing window: {config.testing_window} days")
        self.logger.info(f"📅 Step size: {config.step_size} days")
    
    def run_analysis(self, 
                    market_data: pd.DataFrame,
                    strategy_function: Optional[Callable] = None,
                    benchmark_data: Optional[pd.DataFrame] = None) -> WalkForwardResult:
        """
        Run comprehensive walk-forward analysis.
        
        Args:
            market_data: Historical market data (OHLCV)
            strategy_function: Optional custom strategy function
            benchmark_data: Optional benchmark data for comparison
            
        Returns:
            Walk-forward analysis result
        """
        self.logger.info("🚀 Starting walk-forward analysis")
        start_time = datetime.now()
        
        try:
            # Validate data
            self._validate_data(market_data)
            
            # Reset state
            self._reset_analysis_state()
            
            # Generate walk-forward periods
            periods = self._generate_walk_forward_periods(market_data)
            self.logger.info(f"📊 Generated {len(periods)} walk-forward periods")
            
            # Run analysis for each period
            for i, period in enumerate(periods):
                self.logger.info(f"🔄 Processing period {i+1}/{len(periods)}: {period['start']} to {period['end']}")
                
                # Extract period data
                training_data = period['training_data']
                testing_data = period['testing_data']
                
                # Run backtesting for this period
                period_result = self._analyze_period(
                    training_data, testing_data, strategy_function, benchmark_data
                )
                
                # Store results
                self.period_results.append(period_result)
                
                # Update cumulative metrics
                self._update_cumulative_metrics(period_result)
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics()
            
            # Analyze regime performance
            regime_analysis = self._analyze_regime_performance()
            
            # Create comprehensive result
            result = WalkForwardResult(
                # Overall metrics
                n_periods=len(periods),
                successful_periods=overall_metrics['successful_periods'],
                failed_periods=overall_metrics['failed_periods'],
                success_rate=overall_metrics['success_rate'],
                
                # Performance metrics
                average_return=overall_metrics['average_return'],
                average_sharpe=overall_metrics['average_sharpe'],
                average_drawdown=overall_metrics['average_drawdown'],
                total_return=overall_metrics['total_return'],
                cumulative_return=overall_metrics['cumulative_return'],
                
                # Risk metrics
                volatility=overall_metrics['volatility'],
                max_drawdown=overall_metrics['max_drawdown'],
                var_95=overall_metrics['var_95'],
                cvar_95=overall_metrics['cvar_95'],
                
                # Period-by-period results
                period_results=self.period_results,
                period_returns=overall_metrics['period_returns'],
                period_sharpe=overall_metrics['period_sharpe'],
                period_drawdown=overall_metrics['period_drawdown'],
                
                # Regime analysis
                regime_performance=regime_analysis['regime_performance'],
                regime_stability=regime_analysis['regime_stability'],
                
                # Time series
                equity_curve=pd.Series(self.equity_curve),
                returns_series=pd.Series(self.returns_series),
                drawdown_series=pd.Series(self._calculate_drawdown_series()),
                
                # Metadata
                analysis_period=(market_data.index[0], market_data.index[-1]),
                execution_time=(datetime.now() - start_time).total_seconds(),
                config=self.config
            )
            
            # Save results if configured
            if self.config.save_summary:
                self._save_results(result)
            
            self.results = result
            self.logger.info(f"✅ Walk-forward analysis completed in {result.execution_time:.2f}s")
            self.logger.info(f"📊 Success rate: {result.success_rate:.2%}")
            self.logger.info(f"📈 Average Sharpe: {result.average_sharpe:.3f}")
            self.logger.info(f"📉 Max drawdown: {result.max_drawdown:.2%}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Walk-forward analysis failed: {e}")
            raise
    
    def _validate_data(self, market_data: pd.DataFrame):
        """Validate market data for walk-forward analysis."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in market_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        min_required_samples = self.config.training_window + self.config.testing_window
        if len(market_data) < min_required_samples:
            raise ValueError(f"Insufficient data: {len(market_data)} < {min_required_samples}")
        
        self.logger.info(f"✅ Data validation passed: {len(market_data)} data points")
    
    def _reset_analysis_state(self):
        """Reset analysis state."""
        self.period_results = []
        self.equity_curve = []
        self.returns_series = []
    
    def _generate_walk_forward_periods(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate walk-forward analysis periods."""
        periods = []
        n_samples = len(market_data)
        
        start_idx = 0
        while start_idx + self.config.training_window + self.config.testing_window <= n_samples:
            # Training period
            training_end_idx = start_idx + self.config.training_window
            training_data = market_data.iloc[start_idx:training_end_idx]
            
            # Testing period
            testing_start_idx = training_end_idx
            testing_end_idx = testing_start_idx + self.config.testing_window
            testing_data = market_data.iloc[testing_start_idx:testing_end_idx]
            
            # Create period
            period = {
                'start': market_data.index[start_idx],
                'end': market_data.index[testing_end_idx - 1],
                'training_start': market_data.index[start_idx],
                'training_end': market_data.index[training_end_idx - 1],
                'testing_start': market_data.index[testing_start_idx],
                'testing_end': market_data.index[testing_end_idx - 1],
                'training_data': training_data,
                'testing_data': testing_data
            }
            
            periods.append(period)
            
            # Move to next period
            if self.config.mode == WalkForwardMode.ROLLING:
                start_idx += self.config.step_size
            elif self.config.mode == WalkForwardMode.EXPANDING:
                start_idx += self.config.step_size
            else:  # FIXED
                start_idx += self.config.step_size
        
        return periods
    
    def _analyze_period(self, 
                       training_data: pd.DataFrame,
                       testing_data: pd.DataFrame,
                       strategy_function: Optional[Callable],
                       benchmark_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze a single walk-forward period."""
        try:
            # Create backtesting config for this period
            backtesting_config = BacktestingConfig(
                start_date=testing_data.index[0],
                end_date=testing_data.index[-1],
                initial_capital=100000.0,
                commission_rate=0.001,
                slippage_rate=0.0005
            )
            
            # Initialize backtesting engine
            backtesting_engine = BacktestingEngine(backtesting_config)
            
            # Run backtesting
            result = backtesting_engine.run_backtest(
                market_data=testing_data,
                strategy_function=strategy_function,
                benchmark_data=benchmark_data
            )
            
            # Extract key metrics
            period_result = {
                'period_start': testing_data.index[0],
                'period_end': testing_data.index[-1],
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'max_drawdown': result.max_drawdown,
                'calmar_ratio': result.calmar_ratio,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'var_95': result.var_95,
                'cvar_95': result.cvar_95,
                'success': self._is_period_successful(result),
                'equity_curve': result.equity_curve,
                'returns_series': result.returns_series
            }
            
            return period_result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Period analysis failed: {e}")
            return {
                'period_start': testing_data.index[0],
                'period_end': testing_data.index[-1],
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def _is_period_successful(self, result: BacktestingResult) -> bool:
        """Check if a period is considered successful."""
        return (result.sharpe_ratio >= self.config.min_sharpe_ratio and
                result.max_drawdown >= -self.config.max_drawdown_threshold and
                result.win_rate >= self.config.min_win_rate)
    
    def _update_cumulative_metrics(self, period_result: Dict[str, Any]):
        """Update cumulative metrics with period result."""
        # Update equity curve
        if 'equity_curve' in period_result:
            equity_curve = period_result['equity_curve']
            if len(self.equity_curve) == 0:
                self.equity_curve = equity_curve.tolist()
            else:
                # Append new equity values
                self.equity_curve.extend(equity_curve.tolist())
        
        # Update returns series
        if 'returns_series' in period_result:
            returns_series = period_result['returns_series']
            if len(self.returns_series) == 0:
                self.returns_series = returns_series.tolist()
            else:
                # Append new returns
                self.returns_series.extend(returns_series.tolist())
    
    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall walk-forward metrics."""
        if not self.period_results:
            return {
                'successful_periods': 0,
                'failed_periods': 0,
                'success_rate': 0.0,
                'average_return': 0.0,
                'average_sharpe': 0.0,
                'average_drawdown': 0.0,
                'total_return': 0.0,
                'cumulative_return': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'period_returns': [],
                'period_sharpe': [],
                'period_drawdown': []
            }
        
        # Calculate basic metrics
        successful_periods = sum(1 for result in self.period_results if result.get('success', False))
        failed_periods = len(self.period_results) - successful_periods
        success_rate = successful_periods / len(self.period_results)
        
        # Calculate performance metrics
        period_returns = [result.get('total_return', 0) for result in self.period_results]
        period_sharpe = [result.get('sharpe_ratio', 0) for result in self.period_results]
        period_drawdown = [result.get('max_drawdown', 0) for result in self.period_results]
        
        average_return = np.mean(period_returns)
        average_sharpe = np.mean(period_sharpe)
        average_drawdown = np.mean(period_drawdown)
        
        # Calculate cumulative metrics
        if self.returns_series:
            total_return = (1 + np.array(self.returns_series)).prod() - 1
            cumulative_return = total_return
            volatility = np.std(self.returns_series) * np.sqrt(252)
            
            # Calculate drawdown
            equity_series = pd.Series(self.equity_curve)
            running_max = equity_series.expanding().max()
            drawdown_series = (equity_series - running_max) / running_max
            max_drawdown = drawdown_series.min()
            
            # Calculate VaR and CVaR
            var_95 = np.percentile(self.returns_series, 5)
            cvar_95 = np.mean([r for r in self.returns_series if r <= var_95])
        else:
            total_return = 0.0
            cumulative_return = 0.0
            volatility = 0.0
            max_drawdown = 0.0
            var_95 = 0.0
            cvar_95 = 0.0
        
        return {
            'successful_periods': successful_periods,
            'failed_periods': failed_periods,
            'success_rate': success_rate,
            'average_return': average_return,
            'average_sharpe': average_sharpe,
            'average_drawdown': average_drawdown,
            'total_return': total_return,
            'cumulative_return': cumulative_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'period_returns': period_returns,
            'period_sharpe': period_sharpe,
            'period_drawdown': period_drawdown
        }
    
    def _analyze_regime_performance(self) -> Dict[str, Any]:
        """Analyze regime-specific performance."""
        regime_performance = {}
        regime_stability = {}
        
        # Group results by regime (simplified)
        for result in self.period_results:
            # This would be more sophisticated in practice
            regime_key = "default_regime"
            
            if regime_key not in regime_performance:
                regime_performance[regime_key] = []
                regime_stability[regime_key] = []
            
            regime_performance[regime_key].append(result.get('total_return', 0))
            regime_stability[regime_key].append(result.get('sharpe_ratio', 0))
        
        # Calculate regime metrics
        for regime_key in regime_performance:
            if regime_performance[regime_key]:
                regime_performance[regime_key] = np.mean(regime_performance[regime_key])
                regime_stability[regime_key] = np.std(regime_stability[regime_key])
        
        return {
            'regime_performance': regime_performance,
            'regime_stability': regime_stability
        }
    
    def _calculate_drawdown_series(self) -> List[float]:
        """Calculate drawdown series."""
        if not self.equity_curve:
            return []
        
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown_series = (equity_series - running_max) / running_max
        
        return drawdown_series.tolist()
    
    def _save_results(self, result: WalkForwardResult):
        """Save walk-forward analysis results."""
        try:
            results_dir = Path(self.config.results_directory)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = results_dir / f"walk_forward_summary_{timestamp}.json"
            
            summary = {
                'n_periods': result.n_periods,
                'success_rate': result.success_rate,
                'average_sharpe': result.average_sharpe,
                'max_drawdown': result.max_drawdown,
                'total_return': result.total_return,
                'execution_time': result.execution_time
            }
            
            import json
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save detailed results
            if self.config.save_individual_results:
                details_file = results_dir / f"walk_forward_details_{timestamp}.json"
                
                details = {
                    'period_results': [
                        {
                            'period_start': str(result.period_start),
                            'period_end': str(result.period_end),
                            'total_return': result.total_return,
                            'sharpe_ratio': result.sharpe_ratio,
                            'max_drawdown': result.max_drawdown,
                            'success': result.success
                        }
                        for result in result.period_results
                    ],
                    'config': {
                        'training_window': self.config.training_window,
                        'testing_window': self.config.testing_window,
                        'step_size': self.config.step_size,
                        'mode': self.config.mode.value
                    }
                }
                
                with open(details_file, 'w') as f:
                    json.dump(details, f, indent=2, default=str)
            
            self.logger.info(f"📁 Walk-forward results saved to {results_dir}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save results: {e}")
    
    def get_results(self) -> Optional[WalkForwardResult]:
        """Get walk-forward analysis results."""
        return self.results
    
    def export_results(self, filepath: str):
        """Export results to file."""
        if self.results is None:
            self.logger.warning("⚠️ No results to export")
            return
        
        try:
            # Export equity curve
            equity_file = filepath.replace('.csv', '_equity_curve.csv')
            self.results.equity_curve.to_csv(equity_file)
            
            # Export returns series
            returns_file = filepath.replace('.csv', '_returns.csv')
            self.results.returns_series.to_csv(returns_file)
            
            # Export period results
            periods_file = filepath.replace('.csv', '_periods.csv')
            periods_df = pd.DataFrame(self.results.period_results)
            periods_df.to_csv(periods_file, index=False)
            
            self.logger.info(f"📁 Walk-forward results exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export results: {e}")