"""
Backtesting Engine for TAS

Comprehensive backtesting engine for tree architecture search including
historical data backtesting, performance analysis, and risk assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import TAS components
from src.utils.nas_tas.unsupervised_regime_detection import UnsupervisedRegimeDetector, RegimeDetectionConfig
from ..regime_analysis.regime_qualification import RegimeQualifier, RegimeQualificationConfig
from ..trading.trading_engine import TradingEngine, TradingConfig, TradingResult

logger = logging.getLogger(__name__)


class BacktestingMode(Enum):
    """Backtesting modes."""
    HISTORICAL = "historical"
    WALK_FORWARD = "walk_forward"
    OUT_OF_SAMPLE = "out_of_sample"
    MONTE_CARLO = "monte_carlo"
    SCENARIO = "scenario"


@dataclass
class BacktestingConfig:
    """Configuration for backtesting engine."""
    
    # Backtesting parameters
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    
    # Data parameters
    data_frequency: str = "1H"  # 1H, 1D, etc.
    min_data_points: int = 1000
    max_data_points: int = 10000
    
    # Regime detection parameters
    regime_detection_config: RegimeDetectionConfig = field(default_factory=RegimeDetectionConfig)
    regime_qualification_config: RegimeQualificationConfig = field(default_factory=RegimeQualificationConfig)
    
    # Trading parameters
    trading_config: TradingConfig = field(default_factory=TradingConfig)
    
    # Performance analysis
    benchmark_symbol: Optional[str] = None
    risk_free_rate: float = 0.02  # 2% annual
    
    # Output parameters
    save_results: bool = True
    results_directory: str = "backtesting_results"
    detailed_logging: bool = True
    
    # Advanced parameters
    enable_regime_aware_backtesting: bool = True
    enable_transaction_costs: bool = True
    enable_slippage: bool = True
    enable_market_impact: bool = False


@dataclass
class BacktestingResult:
    """Result of backtesting analysis."""
    
    # Basic metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    
    # Risk metrics
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    cvar_95: float  # Conditional Value at Risk 95%
    cvar_99: float  # Conditional Value at Risk 99%
    beta: float
    alpha: float
    
    # Regime-specific metrics
    regime_performance: Dict[str, float]
    regime_trades: Dict[str, int]
    regime_returns: Dict[str, float]
    
    # Time series data
    equity_curve: pd.Series
    returns_series: pd.Series
    drawdown_series: pd.Series
    
    # Metadata
    backtesting_period: Tuple[datetime, datetime]
    execution_time: float
    config: BacktestingConfig
    
    # Additional analysis
    performance_attribution: Optional[Dict[str, Any]] = None
    risk_analysis: Optional[Dict[str, Any]] = None
    regime_analysis: Optional[Dict[str, Any]] = None


class BacktestingEngine:
    """
    Comprehensive backtesting engine for TAS.
    
    Provides historical backtesting, walk-forward analysis, out-of-sample testing,
    performance attribution, risk analysis, and scenario testing capabilities.
    """
    
    def __init__(self, config: BacktestingConfig):
        """Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.regime_detector = UnsupervisedRegimeDetector(config.regime_detection_config)
        self.regime_qualifier = RegimeQualifier(config.regime_qualification_config)
        self.trading_engine = TradingEngine(config.trading_config)
        
        # Backtesting state
        self.results = None
        self.equity_curve = None
        self.returns_series = None
        self.trades = []
        self.regime_history = []
        
        # Performance tracking
        self.peak_equity = config.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        self.logger.info("✅ Backtesting Engine initialized")
        self.logger.info(f"📅 Backtesting period: {config.start_date} to {config.end_date}")
        self.logger.info(f"💰 Initial capital: ${config.initial_capital:,.2f}")
    
    def run_backtest(self, 
                    market_data: pd.DataFrame,
                    strategy_function: Optional[Callable] = None,
                    benchmark_data: Optional[pd.DataFrame] = None) -> BacktestingResult:
        """
        Run comprehensive backtesting analysis.
        
        Args:
            market_data: Historical market data (OHLCV)
            strategy_function: Optional custom strategy function
            benchmark_data: Optional benchmark data for comparison
            
        Returns:
            Backtesting result with comprehensive metrics
        """
        self.logger.info("🚀 Starting comprehensive backtesting analysis")
        start_time = datetime.now()
        
        try:
            # Validate data
            self._validate_data(market_data)
            
            # Reset state
            self._reset_backtesting_state()
            
            # Step 1: Regime detection and qualification
            self.logger.info("🔍 Step 1: Detecting and qualifying regimes...")
            regime_results = self._detect_and_qualify_regimes(market_data)
            
            # Step 2: Run backtesting simulation
            self.logger.info("📈 Step 2: Running backtesting simulation...")
            simulation_results = self._run_simulation(market_data, regime_results, strategy_function)
            
            # Step 3: Calculate performance metrics
            self.logger.info("📊 Step 3: Calculating performance metrics...")
            performance_metrics = self._calculate_performance_metrics(simulation_results, benchmark_data)
            
            # Step 4: Risk analysis
            self.logger.info("⚠️ Step 4: Performing risk analysis...")
            risk_metrics = self._calculate_risk_metrics(simulation_results)
            
            # Step 5: Regime-specific analysis
            self.logger.info("🎯 Step 5: Analyzing regime-specific performance...")
            regime_analysis = self._analyze_regime_performance(simulation_results, regime_results)
            
            # Step 6: Performance attribution
            self.logger.info("🔍 Step 6: Calculating performance attribution...")
            attribution_analysis = self._calculate_performance_attribution(simulation_results, regime_results)
            
            # Create comprehensive result
            result = BacktestingResult(
                # Basic metrics
                total_return=performance_metrics['total_return'],
                annualized_return=performance_metrics['annualized_return'],
                volatility=performance_metrics['volatility'],
                sharpe_ratio=performance_metrics['sharpe_ratio'],
                sortino_ratio=performance_metrics['sortino_ratio'],
                max_drawdown=performance_metrics['max_drawdown'],
                calmar_ratio=performance_metrics['calmar_ratio'],
                
                # Trading metrics
                total_trades=simulation_results['total_trades'],
                winning_trades=simulation_results['winning_trades'],
                losing_trades=simulation_results['losing_trades'],
                win_rate=simulation_results['win_rate'],
                profit_factor=simulation_results['profit_factor'],
                average_win=simulation_results['average_win'],
                average_loss=simulation_results['average_loss'],
                
                # Risk metrics
                var_95=risk_metrics['var_95'],
                var_99=risk_metrics['var_99'],
                cvar_95=risk_metrics['cvar_95'],
                cvar_99=risk_metrics['cvar_99'],
                beta=risk_metrics['beta'],
                alpha=risk_metrics['alpha'],
                
                # Regime-specific metrics
                regime_performance=regime_analysis['regime_performance'],
                regime_trades=regime_analysis['regime_trades'],
                regime_returns=regime_analysis['regime_returns'],
                
                # Time series data
                equity_curve=simulation_results['equity_curve'],
                returns_series=simulation_results['returns_series'],
                drawdown_series=simulation_results['drawdown_series'],
                
                # Metadata
                backtesting_period=(self.config.start_date, self.config.end_date),
                execution_time=(datetime.now() - start_time).total_seconds(),
                config=self.config,
                
                # Additional analysis
                performance_attribution=attribution_analysis,
                risk_analysis=risk_metrics,
                regime_analysis=regime_analysis
            )
            
            # Save results if configured
            if self.config.save_results:
                self._save_results(result)
            
            self.results = result
            self.logger.info(f"✅ Backtesting completed in {result.execution_time:.2f}s")
            self.logger.info(f"📊 Total return: {result.total_return:.2%}")
            self.logger.info(f"📈 Sharpe ratio: {result.sharpe_ratio:.3f}")
            self.logger.info(f"📉 Max drawdown: {result.max_drawdown:.2%}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Backtesting failed: {e}")
            raise
    
    def _validate_data(self, market_data: pd.DataFrame):
        """Validate market data for backtesting."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in market_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if len(market_data) < self.config.min_data_points:
            raise ValueError(f"Insufficient data points: {len(market_data)} < {self.config.min_data_points}")
        
        if len(market_data) > self.config.max_data_points:
            self.logger.warning(f"Large dataset: {len(market_data)} points, using first {self.config.max_data_points}")
            market_data = market_data.head(self.config.max_data_points)
        
        # Check for missing values
        missing_values = market_data[required_columns].isnull().sum()
        if missing_values.any():
            self.logger.warning(f"Missing values detected: {missing_values.to_dict()}")
        
        self.logger.info(f"✅ Data validation passed: {len(market_data)} data points")
    
    def _reset_backtesting_state(self):
        """Reset backtesting state."""
        self.trading_engine.reset_trading_state()
        self.equity_curve = []
        self.returns_series = []
        self.trades = []
        self.regime_history = []
        self.peak_equity = self.config.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
    
    def _detect_and_qualify_regimes(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect and qualify regimes in the data."""
        # Detect regimes
        regime_results = self.regime_detector.detect_regimes(market_data)
        
        # Qualify regimes
        qualification_results = self.regime_qualifier.qualify_regimes(regime_results, market_data)
        
        return {
            'regime_results': regime_results,
            'qualification_results': qualification_results,
            'qualified_regimes': qualification_results.get('qualified_regimes', {}),
            'regime_labels': regime_results.get('regime_labels', np.array([]))
        }
    
    def _run_simulation(self, 
                       market_data: pd.DataFrame, 
                       regime_results: Dict[str, Any],
                       strategy_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Run backtesting simulation."""
        equity_curve = [self.config.initial_capital]
        returns_series = []
        trades = []
        regime_history = []
        
        # Get regime labels
        regime_labels = regime_results.get('regime_labels', np.array([]))
        qualified_regimes = regime_results.get('qualified_regimes', {})
        
        # Simulate trading over time
        for i in range(1, len(market_data)):
            current_data = market_data.iloc[:i+1]
            current_price = market_data.iloc[i]['close']
            current_regime = regime_labels[i] if i < len(regime_labels) else None
            
            # Get regime information
            regime_info = None
            if current_regime is not None:
                for regime_name, regime_data in qualified_regimes.items():
                    if regime_data.get('regime_id') == current_regime:
                        regime_info = regime_data
                        break
            
            # Generate trading signals
            if strategy_function:
                signals = strategy_function(current_data, regime_info)
            else:
                signals = self._default_strategy(current_data, regime_info)
            
            # Execute trades
            if signals:
                for signal in signals:
                    trade_result = self._execute_trade(signal, current_price, i)
                    if trade_result:
                        trades.append(trade_result)
            
            # Update equity curve
            current_equity = self.trading_engine.get_current_capital()
            equity_curve.append(current_equity)
            
            # Calculate returns
            if len(equity_curve) > 1:
                period_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                returns_series.append(period_return)
            else:
                returns_series.append(0.0)
            
            # Track regime history
            regime_history.append({
                'timestamp': market_data.index[i] if hasattr(market_data, 'index') else i,
                'regime_id': current_regime,
                'regime_info': regime_info,
                'equity': current_equity
            })
        
        # Calculate trading metrics
        trading_metrics = self._calculate_trading_metrics(trades)
        
        return {
            'equity_curve': pd.Series(equity_curve, index=market_data.index[:len(equity_curve)]),
            'returns_series': pd.Series(returns_series, index=market_data.index[1:len(returns_series)+1]),
            'trades': trades,
            'regime_history': regime_history,
            'trading_metrics': trading_metrics
        }
    
    def _default_strategy(self, market_data: pd.DataFrame, regime_info: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Default trading strategy based on regime information."""
        signals = []
        
        if not regime_info:
            return signals
        
        # Simple strategy based on regime characteristics
        volatility = regime_info.get('price_volatility', 0)
        trend = regime_info.get('price_trend', 0)
        confidence = regime_info.get('confidence', 0)
        
        # Only trade if regime is qualified and has sufficient confidence
        if confidence > 0.6:
            if trend > 0.02:  # Strong uptrend
                signals.append({
                    'symbol': 'BTC',  # Default symbol
                    'side': 'buy',
                    'quantity': 0.1,  # 10% of capital
                    'price': market_data['close'].iloc[-1],
                    'regime_info': regime_info
                })
            elif trend < -0.02:  # Strong downtrend
                signals.append({
                    'symbol': 'BTC',
                    'side': 'sell',
                    'quantity': 0.1,
                    'price': market_data['close'].iloc[-1],
                    'regime_info': regime_info
                })
        
        return signals
    
    def _execute_trade(self, signal: Dict[str, Any], current_price: float, timestamp: int) -> Optional[Dict[str, Any]]:
        """Execute a trade signal."""
        try:
            # Convert signal to trade execution
            side = signal['side']
            quantity = signal['quantity']
            price = signal.get('price', current_price)
            regime_info = signal.get('regime_info')
            
            # Execute trade through trading engine
            trade_result = self.trading_engine.execute_trade(
                symbol=signal.get('symbol', 'BTC'),
                side=side,
                quantity=quantity,
                price=price,
                regime_info=regime_info
            )
            
            return trade_result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trade execution failed: {e}")
            return None
    
    def _calculate_trading_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trading metrics from executed trades."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0
            }
        
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        losing_trades = sum(1 for trade in trades if trade.get('pnl', 0) < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit factor
        total_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
        total_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate average win/loss
        average_win = total_profit / winning_trades if winning_trades > 0 else 0.0
        average_loss = total_loss / losing_trades if losing_trades > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': average_win,
            'average_loss': average_loss
        }
    
    def _calculate_performance_metrics(self, 
                                     simulation_results: Dict[str, Any],
                                     benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        equity_curve = simulation_results['equity_curve']
        returns_series = simulation_results['returns_series']
        
        # Basic return metrics
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0.0
        
        # Volatility
        volatility = returns_series.std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        risk_free_rate = self.config.risk_free_rate
        excess_returns = returns_series.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0.0
        
        # Sortino ratio
        downside_returns = returns_series[returns_series < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = excess_returns / downside_volatility if downside_volatility > 0 else 0.0
        
        # Maximum drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        }
    
    def _calculate_risk_metrics(self, simulation_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        returns_series = simulation_results['returns_series']
        
        if len(returns_series) == 0:
            return {
                'var_95': 0.0, 'var_99': 0.0,
                'cvar_95': 0.0, 'cvar_99': 0.0,
                'beta': 0.0, 'alpha': 0.0
            }
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns_series, 5)
        var_99 = np.percentile(returns_series, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns_series[returns_series <= var_95].mean() if len(returns_series[returns_series <= var_95]) > 0 else var_95
        cvar_99 = returns_series[returns_series <= var_99].mean() if len(returns_series[returns_series <= var_99]) > 0 else var_99
        
        # Beta and Alpha (simplified - would need benchmark data for proper calculation)
        beta = 1.0  # Placeholder
        alpha = 0.0  # Placeholder
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'beta': beta,
            'alpha': alpha
        }
    
    def _analyze_regime_performance(self, 
                                  simulation_results: Dict[str, Any],
                                  regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance by regime."""
        regime_history = simulation_results['regime_history']
        returns_series = simulation_results['returns_series']
        
        regime_performance = {}
        regime_trades = {}
        regime_returns = {}
        
        # Group performance by regime
        for i, regime_data in enumerate(regime_history):
            regime_id = regime_data.get('regime_id')
            if regime_id is not None:
                regime_key = f"regime_{regime_id}"
                
                if regime_key not in regime_performance:
                    regime_performance[regime_key] = []
                    regime_trades[regime_key] = 0
                    regime_returns[regime_key] = []
                
                # Track returns for this regime
                if i < len(returns_series):
                    regime_returns[regime_key].append(returns_series.iloc[i])
        
        # Calculate regime-specific metrics
        for regime_key in regime_performance:
            if regime_returns[regime_key]:
                regime_performance[regime_key] = np.mean(regime_returns[regime_key])
        
        return {
            'regime_performance': regime_performance,
            'regime_trades': regime_trades,
            'regime_returns': regime_returns
        }
    
    def _calculate_performance_attribution(self, 
                                         simulation_results: Dict[str, Any],
                                         regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance attribution analysis."""
        # This would include detailed attribution analysis
        # For now, return basic structure
        return {
            'regime_attribution': {},
            'time_attribution': {},
            'factor_attribution': {}
        }
    
    def _save_results(self, result: BacktestingResult):
        """Save backtesting results to file."""
        try:
            results_dir = Path(self.config.results_directory)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"backtesting_results_{timestamp}.json"
            
            # Convert result to serializable format
            result_dict = {
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
                'backtesting_period': [result.backtesting_period[0].isoformat(), result.backtesting_period[1].isoformat()],
                'execution_time': result.execution_time,
                'config': {
                    'start_date': result.config.start_date.isoformat(),
                    'end_date': result.config.end_date.isoformat(),
                    'initial_capital': result.config.initial_capital,
                    'commission_rate': result.config.commission_rate
                }
            }
            
            with open(results_file, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            self.logger.info(f"📁 Results saved to {results_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save results: {e}")
    
    def get_results(self) -> Optional[BacktestingResult]:
        """Get backtesting results."""
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
            
            # Export summary
            summary_file = filepath.replace('.csv', '_summary.json')
            summary = {
                'total_return': self.results.total_return,
                'sharpe_ratio': self.results.sharpe_ratio,
                'max_drawdown': self.results.max_drawdown,
                'total_trades': self.results.total_trades,
                'win_rate': self.results.win_rate
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"📁 Results exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export results: {e}")