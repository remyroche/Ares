"""
Backtesting-Integrated Labeling Validation

This module provides validation of profit labeling through actual backtesting
performance. It creates trading strategies based on labels and evaluates their
real-world performance to validate labeling effectiveness.

Key Validation Components:
1. Label-Based Strategy Creation
2. Comprehensive Backtesting Engine
3. Performance Analysis and Ranking
4. Risk-Adjusted Metrics
5. Economic Significance Testing
6. Regime-Specific Performance Analysis
7. Transaction Cost Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Performance analysis imports
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

class StrategyType(Enum):
    """Enumeration of strategy types based on labels."""
    THRESHOLD_STRATEGY = "threshold_strategy"
    RANKING_STRATEGY = "ranking_strategy"
    MOMENTUM_STRATEGY = "momentum_strategy"
    MEAN_REVERSION_STRATEGY = "mean_reversion_strategy"
    ENSEMBLE_STRATEGY = "ensemble_strategy"
    ADAPTIVE_STRATEGY = "adaptive_strategy"

class PerformanceMetric(Enum):
    """Enumeration of performance metrics."""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    INFORMATION_RATIO = "information_ratio"
    ALPHA = "alpha"
    BETA = "beta"

class RiskMeasure(Enum):
    """Enumeration of risk measures."""
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VAR = "conditional_var"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    DOWNSIDE_DEVIATION = "downside_deviation"
    ULCER_INDEX = "ulcer_index"
    TAIL_RATIO = "tail_ratio"

@dataclass
class BacktestingConfig:
    """Configuration for backtesting-integrated validation."""
    # Strategy configuration
    strategy_types: List[StrategyType] = field(default_factory=lambda: [
        StrategyType.THRESHOLD_STRATEGY,
        StrategyType.RANKING_STRATEGY,
        StrategyType.MOMENTUM_STRATEGY
    ])

    # Threshold strategy parameters
    threshold_percentiles: List[float] = field(default_factory=lambda: [0.6, 0.7, 0.8, 0.9])

    # Ranking strategy parameters
    ranking_quantiles: List[float] = field(default_factory=lambda: [0.8, 0.9, 0.95])

    # Transaction costs
    transaction_cost: float = 0.0008  # 0.08% per trade
    slippage: float = 0.0002  # 0.02% slippage

    # Position sizing
    position_sizing_method: str = "equal_weight"  # "equal_weight", "volatility_scaled", "kelly"
    max_position_size: float = 0.1  # 10% maximum position

    # Risk management
    stop_loss: Optional[float] = None  # Stop loss percentage
    take_profit: Optional[float] = None  # Take profit percentage
    max_holding_period: Optional[int] = None  # Maximum holding period in periods

    # Backtesting parameters
    initial_capital: float = 100000.0
    rebalance_frequency: int = 1  # Rebalance every N periods
    warmup_period: int = 100  # Periods to warm up indicators

    # Performance analysis
    benchmark_return: float = 0.0  # Risk-free rate or benchmark return
    confidence_level: float = 0.95  # For VaR calculations

    # Validation parameters
    validation_split: float = 0.3
    min_trades: int = 50  # Minimum trades for valid backtest
    min_holding_periods: int = 10  # Minimum different holding periods

    # Regime analysis
    analyze_regime_performance: bool = True
    regime_definition_method: str = "volatility"  # "volatility", "trend", "combined"

    # Parallel processing
    n_jobs: int = -1

@dataclass
class Trade:
    """Container for individual trade information."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    holding_period: int
    strategy_signal: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyPerformance:
    """Container for strategy performance metrics."""
    strategy_name: str
    strategy_type: StrategyType
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    avg_holding_period: float
    performance_metrics: Dict[PerformanceMetric, float]
    risk_metrics: Dict[RiskMeasure, float]
    trades: List[Trade]
    equity_curve: pd.Series
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestingValidationResult:
    """Result container for backtesting validation."""
    strategy_performances: Dict[str, StrategyPerformance]
    label_quality_ranking: Dict[str, float]
    economic_significance_tests: Dict[str, Dict[str, float]]
    regime_performance_analysis: Dict[str, Dict[str, float]]
    comparative_analysis: Dict[str, Any]
    validation_summary: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class LabelBasedStrategy:
    """Base class for label-based trading strategies."""

    def __init__(self, strategy_type: StrategyType, config: BacktestingConfig):
        """Initialize label-based strategy."""
        self.strategy_type = strategy_type
        self.config = config
        self.logger = get_logger(f'Strategy_{strategy_type.value}')

    def generate_signals(self,
                        labels: pd.DataFrame,
                        market_data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from labels."""
        raise NotImplementedError("Subclasses must implement generate_signals")

    def calculate_position_size(self,
                              signal: float,
                              price: float,
                              portfolio_value: float,
                              volatility: Optional[float] = None) -> float:
        """Calculate position size based on signal and risk management."""
        if self.config.position_sizing_method == "equal_weight":
            return signal * self.config.max_position_size * portfolio_value / price

        elif self.config.position_sizing_method == "volatility_scaled" and volatility is not None:
            # Scale position size inversely with volatility
            vol_adjustment = 0.02 / max(volatility, 0.001)  # Target 2% volatility
            adjusted_size = signal * self.config.max_position_size * vol_adjustment
            return min(adjusted_size, self.config.max_position_size) * portfolio_value / price

        else:
            return signal * self.config.max_position_size * portfolio_value / price

class ThresholdStrategy(LabelBasedStrategy):
    """Strategy based on label threshold crossing."""

    def __init__(self, config: BacktestingConfig, threshold: float = 0.7):
        """Initialize threshold strategy."""
        super().__init__(StrategyType.THRESHOLD_STRATEGY, config)
        self.threshold = threshold

    def generate_signals(self,
                        labels: pd.DataFrame,
                        market_data: pd.DataFrame) -> pd.Series:
        """Generate signals based on threshold crossing."""
        # Use main opportunity column
        opportunity_cols = [col for col in labels.columns if 'opportunity' in col.lower()]

        if not opportunity_cols:
            return pd.Series(0.0, index=labels.index)

        main_signal = labels[opportunity_cols[0]].fillna(0)

        # Generate binary signals based on threshold
        signals = (main_signal > self.threshold).astype(float)

        return signals

class RankingStrategy(LabelBasedStrategy):
    """Strategy based on ranking labels and selecting top quantile."""

    def __init__(self, config: BacktestingConfig, quantile: float = 0.8):
        """Initialize ranking strategy."""
        super().__init__(StrategyType.RANKING_STRATEGY, config)
        self.quantile = quantile

    def generate_signals(self,
                        labels: pd.DataFrame,
                        market_data: pd.DataFrame) -> pd.Series:
        """Generate signals based on ranking."""
        # Use main opportunity column
        opportunity_cols = [col for col in labels.columns if 'opportunity' in col.lower()]

        if not opportunity_cols:
            return pd.Series(0.0, index=labels.index)

        main_signal = labels[opportunity_cols[0]].fillna(0)

        # Calculate rolling quantile threshold
        rolling_threshold = main_signal.rolling(100, min_periods=20).quantile(self.quantile)

        # Generate signals for top quantile
        signals = (main_signal > rolling_threshold).astype(float)

        return signals

class MomentumStrategy(LabelBasedStrategy):
    """Strategy based on label momentum."""

    def __init__(self, config: BacktestingConfig, momentum_window: int = 5):
        """Initialize momentum strategy."""
        super().__init__(StrategyType.MOMENTUM_STRATEGY, config)
        self.momentum_window = momentum_window

    def generate_signals(self,
                        labels: pd.DataFrame,
                        market_data: pd.DataFrame) -> pd.Series:
        """Generate signals based on label momentum."""
        # Use main opportunity column
        opportunity_cols = [col for col in labels.columns if 'opportunity' in col.lower()]

        if not opportunity_cols:
            return pd.Series(0.0, index=labels.index)

        main_signal = labels[opportunity_cols[0]].fillna(0)

        # Calculate momentum (change in signal over window)
        momentum = main_signal.diff(self.momentum_window)

        # Generate signals based on positive momentum and high current signal
        signals = ((momentum > 0) & (main_signal > 0.5)).astype(float)

        return signals

class BacktestingEngine:
    """Backtesting engine for strategy validation."""

    def __init__(self, config: BacktestingConfig):
        """Initialize backtesting engine."""
        self.config = config
        self.logger = get_logger('BacktestingEngine')

        # Backtesting state
        self.current_positions: Dict[str, float] = {}
        self.portfolio_value: float = config.initial_capital
        self.cash: float = config.initial_capital
        self.trades: List[Trade] = []

    def backtest_strategy(self,
                         strategy: LabelBasedStrategy,
                         labels: pd.DataFrame,
                         market_data: pd.DataFrame,
                         strategy_name: str) -> StrategyPerformance:
        """Backtest a single strategy."""
        self.logger.info(f'🔄 Backtesting strategy: {strategy_name}')

        # Reset state
        self._reset_backtest_state()

        # Generate signals
        signals = strategy.generate_signals(labels, market_data)

        # Align data
        common_idx = signals.index.intersection(market_data.index)
        if len(common_idx) < self.config.min_trades:
            self.logger.warning(f'⚠️ Insufficient data for {strategy_name}')
            return self._create_empty_performance(strategy_name, strategy.strategy_type)

        aligned_signals = signals.loc[common_idx]
        aligned_market = market_data.loc[common_idx]

        # Run backtest
        equity_curve = self._run_backtest(aligned_signals, aligned_market, strategy)

        # Calculate performance metrics
        performance = self._calculate_performance_metrics(
            strategy_name, strategy.strategy_type, equity_curve, aligned_market
        )

        self.logger.info(f'✅ Backtest completed for {strategy_name}')
        self.logger.info(f'   → Total Return: {performance.total_return:.2%}')
        self.logger.info(f'   → Sharpe Ratio: {performance.sharpe_ratio:.3f}')
        self.logger.info(f'   → Max Drawdown: {performance.max_drawdown:.2%}')

        return performance

    def _reset_backtest_state(self):
        """Reset backtesting state."""
        self.current_positions = {}
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.trades = []

    def _run_backtest(self,
                     signals: pd.Series,
                     market_data: pd.DataFrame,
                     strategy: LabelBasedStrategy) -> pd.Series:
        """Run the actual backtest simulation."""
        equity_curve = []
        prices = market_data['close'] if 'close' in market_data.columns else market_data.iloc[:, 0]

        # Calculate volatility for position sizing
        returns = prices.pct_change()
        volatility = returns.rolling(20).std()

        for i, (timestamp, signal) in enumerate(signals.items()):
            current_price = prices.loc[timestamp]
            current_vol = volatility.loc[timestamp] if not pd.isna(volatility.loc[timestamp]) else 0.02

            # Skip warmup period
            if i < self.config.warmup_period:
                equity_curve.append(self.config.initial_capital)
                continue

            # Current position
            current_position = self.current_positions.get('position', 0.0)

            # Calculate target position
            if signal > 0:
                target_position = strategy.calculate_position_size(
                    signal, current_price, self.portfolio_value, current_vol
                )
            else:
                target_position = 0.0

            # Execute trades if position change is significant
            position_change = target_position - current_position

            if abs(position_change) > 0.01:  # Minimum trade size threshold
                # Calculate transaction costs
                trade_value = abs(position_change * current_price)
                transaction_cost = trade_value * (self.config.transaction_cost + self.config.slippage)

                # Execute trade
                if position_change > 0:  # Buy
                    cost = position_change * current_price + transaction_cost
                    if cost <= self.cash:
                        self.cash -= cost
                        self.current_positions['position'] = target_position

                        # Record trade (entry)
                        if current_position == 0:  # New position
                            self.trades.append(Trade(
                                entry_time=timestamp,
                                exit_time=timestamp,  # Will be updated on exit
                                entry_price=current_price,
                                exit_price=current_price,
                                quantity=position_change,
                                pnl=0.0,
                                pnl_pct=0.0,
                                holding_period=0,
                                strategy_signal=signal,
                                metadata={'type': 'entry'}
                            ))

                else:  # Sell
                    proceeds = abs(position_change) * current_price - transaction_cost
                    self.cash += proceeds
                    self.current_positions['position'] = target_position

                    # Record trade (exit) - update last trade
                    if self.trades and self.trades[-1].metadata.get('type') == 'entry':
                        last_trade = self.trades[-1]
                        last_trade.exit_time = timestamp
                        last_trade.exit_price = current_price
                        last_trade.holding_period = i - signals.index.get_loc(last_trade.entry_time)
                        last_trade.pnl = (current_price - last_trade.entry_price) * last_trade.quantity - transaction_cost
                        last_trade.pnl_pct = last_trade.pnl / (last_trade.entry_price * last_trade.quantity)
                        last_trade.metadata['type'] = 'complete'

            # Update portfolio value
            position_value = self.current_positions.get('position', 0) * current_price
            self.portfolio_value = self.cash + position_value
            equity_curve.append(self.portfolio_value)

        return pd.Series(equity_curve, index=signals.index)

    def _calculate_performance_metrics(self,
                                     strategy_name: str,
                                     strategy_type: StrategyType,
                                     equity_curve: pd.Series,
                                     market_data: pd.DataFrame) -> StrategyPerformance:
        """Calculate comprehensive performance metrics."""
        # Basic returns
        returns = equity_curve.pct_change().dropna()

        if len(returns) == 0:
            return self._create_empty_performance(strategy_name, strategy_type)

        # Total return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Annualized return (assuming 5-minute bars, 288 per day, 252 trading days)
        periods_per_year = 288 * 252
        n_periods = len(equity_curve)
        years = n_periods / periods_per_year
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        volatility = returns.std() * np.sqrt(periods_per_year)

        # Sharpe ratio
        excess_returns = returns - self.config.benchmark_return / periods_per_year
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else 0

        # Maximum drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Trade statistics
        completed_trades = [t for t in self.trades if t.metadata.get('type') == 'complete']

        if completed_trades:
            win_rate = sum(1 for t in completed_trades if t.pnl > 0) / len(completed_trades)
            avg_trade_return = np.mean([t.pnl_pct for t in completed_trades])
            avg_holding_period = np.mean([t.holding_period for t in completed_trades])

            # Profit factor
            winning_trades = [t.pnl for t in completed_trades if t.pnl > 0]
            losing_trades = [t.pnl for t in completed_trades if t.pnl < 0]

            total_wins = sum(winning_trades) if winning_trades else 0
            total_losses = abs(sum(losing_trades)) if losing_trades else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            win_rate = 0
            avg_trade_return = 0
            avg_holding_period = 0
            profit_factor = 0

        # Performance metrics dictionary
        performance_metrics = {
            PerformanceMetric.TOTAL_RETURN: total_return,
            PerformanceMetric.SHARPE_RATIO: sharpe_ratio,
            PerformanceMetric.SORTINO_RATIO: sortino_ratio,
            PerformanceMetric.CALMAR_RATIO: calmar_ratio,
            PerformanceMetric.MAX_DRAWDOWN: max_drawdown,
            PerformanceMetric.VOLATILITY: volatility,
            PerformanceMetric.WIN_RATE: win_rate,
            PerformanceMetric.PROFIT_FACTOR: profit_factor
        }

        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(returns, equity_curve)

        return StrategyPerformance(
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(completed_trades),
            avg_trade_return=avg_trade_return,
            avg_holding_period=avg_holding_period,
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            trades=completed_trades,
            equity_curve=equity_curve,
            metadata={'periods_per_year': periods_per_year, 'years': years}
        )

    def _calculate_risk_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> Dict[RiskMeasure, float]:
        """Calculate risk metrics using VectorBT."""
        risk_metrics = {}

        if len(returns) == 0:
            return risk_metrics

        try:
            import vectorbt as vbt
            from vectorbt.returns import Returns
            from vectorbt.portfolio import Portfolio

            # Use VectorBT Returns for risk calculations
            returns_obj = Returns(returns)

            # Value at Risk (VaR) using VectorBT
            var_95 = returns_obj.var(alpha=1 - self.config.confidence_level)
            risk_metrics[RiskMeasure.VALUE_AT_RISK] = abs(var_95)

            # Conditional VaR (Expected Shortfall) using VectorBT
            cvar_95 = returns_obj.cvar(alpha=1 - self.config.confidence_level)
            risk_metrics[RiskMeasure.CONDITIONAL_VAR] = abs(cvar_95)

            # Maximum Drawdown using VectorBT
            portfolio = Portfolio.from_returns(returns)
            max_dd = portfolio.max_drawdown()
            risk_metrics[RiskMeasure.MAXIMUM_DRAWDOWN] = abs(max_dd)

            # Downside Deviation using VectorBT
            downside_deviation = returns_obj.downside_deviation()
            risk_metrics[RiskMeasure.DOWNSIDE_DEVIATION] = downside_deviation

            # Ulcer Index using VectorBT
            ulcer_index = returns_obj.ulcer_index()
            risk_metrics[RiskMeasure.ULCER_INDEX] = ulcer_index

            # Tail Ratio using VectorBT
            tail_ratio = returns_obj.tail_ratio()
            risk_metrics[RiskMeasure.TAIL_RATIO] = tail_ratio

        except Exception as e:
            self.logger.warning(f'VectorBT risk metrics calculation failed, using manual calculation: {e}')
            # Fallback to manual calculation
            try:
                # Value at Risk (VaR)
                var_95 = np.percentile(returns, (1 - self.config.confidence_level) * 100)
                risk_metrics[RiskMeasure.VALUE_AT_RISK] = abs(var_95)

                # Conditional VaR (Expected Shortfall)
                cvar_95 = returns[returns <= var_95].mean()
                risk_metrics[RiskMeasure.CONDITIONAL_VAR] = abs(cvar_95)

                # Maximum Drawdown
                rolling_max = equity_curve.expanding().max()
                drawdown = (equity_curve - rolling_max) / rolling_max
                risk_metrics[RiskMeasure.MAXIMUM_DRAWDOWN] = abs(drawdown.min())

                # Downside Deviation
                downside_returns = returns[returns < returns.mean()]
                if len(downside_returns) > 0:
                    downside_deviation = downside_returns.std()
                    risk_metrics[RiskMeasure.DOWNSIDE_DEVIATION] = downside_deviation

                # Ulcer Index
                squared_drawdowns = drawdown ** 2
                ulcer_index = np.sqrt(squared_drawdowns.mean())
                risk_metrics[RiskMeasure.ULCER_INDEX] = ulcer_index

                # Tail Ratio
                p95 = np.percentile(returns, 95)
                p5 = np.percentile(returns, 5)
                if p5 != 0:
                    tail_ratio = p95 / abs(p5)
                    risk_metrics[RiskMeasure.TAIL_RATIO] = tail_ratio

            except Exception as e2:
                self.logger.warning(f'Manual risk metrics calculation also failed: {e2}')

        return risk_metrics

    def _create_empty_performance(self, strategy_name: str, strategy_type: StrategyType) -> StrategyPerformance:
        """Create empty performance result."""
        return StrategyPerformance(
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            avg_trade_return=0.0,
            avg_holding_period=0.0,
            performance_metrics={},
            risk_metrics={},
            trades=[],
            equity_curve=pd.Series(dtype=float),
            metadata={'error': 'insufficient_data'}
        )

class BacktestingIntegratedValidator:
    """
    Main validator that uses backtesting to validate labeling quality.

    This class creates trading strategies based on labels and evaluates their
    performance to determine the economic value and practical utility of the labels.
    """

    def __init__(self, config: Optional[BacktestingConfig] = None):
        """Initialize backtesting-integrated validator."""
        self.config = config or BacktestingConfig()
        self.logger = get_logger('BacktestingIntegratedValidator')

        # Validation components
        self.backtesting_engine = BacktestingEngine(self.config)

        # Results storage
        self.validation_results: Optional[BacktestingValidationResult] = None

        self.logger.info('🎯 Backtesting-Integrated Validator initialized')
        self.logger.info(f'   → Strategy types: {[s.value for s in self.config.strategy_types]}')

    def validate_through_backtesting(self,
                                   labeled_data: pd.DataFrame,
                                   market_data: pd.DataFrame) -> BacktestingValidationResult:
        """
        Validate labels through comprehensive backtesting.

        Args:
            labeled_data: DataFrame with profit labels
            market_data: OHLCV market data

        Returns:
            BacktestingValidationResult with comprehensive analysis
        """
        self.logger.info('🚀 Starting backtesting-integrated validation')

        # Split data for validation
        split_idx = int(len(labeled_data) * (1 - self.config.validation_split))

        train_labels = labeled_data.iloc[:split_idx]
        train_market = market_data.iloc[:split_idx]
        val_labels = labeled_data.iloc[split_idx:]
        val_market = market_data.iloc[split_idx:]

        # Create and test strategies
        strategies = self._create_strategies()
        strategy_performances = {}

        for strategy_name, strategy in strategies.items():
            try:
                performance = self.backtesting_engine.backtest_strategy(
                    strategy, val_labels, val_market, strategy_name
                )
                strategy_performances[strategy_name] = performance

            except Exception as e:
                self.logger.error(f'Strategy {strategy_name} failed: {e}')

        # Analyze results
        label_quality_ranking = self._rank_label_quality(strategy_performances)
        economic_significance = self._test_economic_significance(strategy_performances)
        regime_analysis = self._analyze_regime_performance(strategy_performances, val_market)
        comparative_analysis = self._perform_comparative_analysis(strategy_performances)
        validation_summary = self._generate_validation_summary(strategy_performances)

        # Create result
        self.validation_results = BacktestingValidationResult(
            strategy_performances=strategy_performances,
            label_quality_ranking=label_quality_ranking,
            economic_significance_tests=economic_significance,
            regime_performance_analysis=regime_analysis,
            comparative_analysis=comparative_analysis,
            validation_summary=validation_summary,
            metadata={
                'n_strategies': len(strategy_performances),
                'validation_period': len(val_labels),
                'total_trades': sum(p.total_trades for p in strategy_performances.values())
            }
        )

        self.logger.info('✅ Backtesting validation completed')
        self.logger.info(f'   → Tested {len(strategy_performances)} strategies')
        self.logger.info(f'   → Best strategy: {max(label_quality_ranking.items(), key=lambda x: x[1])[0]}')

        return self.validation_results

    def _create_strategies(self) -> Dict[str, LabelBasedStrategy]:
        """Create strategies for testing."""
        strategies = {}

        for strategy_type in self.config.strategy_types:
            if strategy_type == StrategyType.THRESHOLD_STRATEGY:
                for threshold in self.config.threshold_percentiles:
                    strategy_name = f"threshold_{threshold:.1f}"
                    strategies[strategy_name] = ThresholdStrategy(self.config, threshold)

            elif strategy_type == StrategyType.RANKING_STRATEGY:
                for quantile in self.config.ranking_quantiles:
                    strategy_name = f"ranking_{quantile:.2f}"
                    strategies[strategy_name] = RankingStrategy(self.config, quantile)

            elif strategy_type == StrategyType.MOMENTUM_STRATEGY:
                for window in [3, 5, 10]:
                    strategy_name = f"momentum_{window}"
                    strategies[strategy_name] = MomentumStrategy(self.config, window)

        return strategies

    def _rank_label_quality(self,
                           strategy_performances: Dict[str, StrategyPerformance]) -> Dict[str, float]:
        """Rank label quality based on strategy performance."""
        quality_scores = {}

        if not strategy_performances:
            return quality_scores

        # Calculate composite quality score for each strategy
        for strategy_name, performance in strategy_performances.items():
            # Multi-factor quality score
            sharpe_score = max(0, min(2, performance.sharpe_ratio + 1)) / 2  # Normalize to 0-1
            return_score = max(0, min(1, performance.total_return + 0.5)) if performance.total_return > -0.5 else 0
            drawdown_penalty = performance.max_drawdown  # Penalty for high drawdown
            trade_bonus = min(1, performance.total_trades / 100)  # Bonus for sufficient trades

            # Composite score
            composite_score = (
                0.4 * sharpe_score +
                0.3 * return_score +
                0.2 * trade_bonus -
                0.1 * drawdown_penalty
            )

            quality_scores[strategy_name] = max(0, composite_score)

        return quality_scores

    def _test_economic_significance(self,
                                  strategy_performances: Dict[str, StrategyPerformance]) -> Dict[str, Dict[str, float]]:
        """Test economic significance of strategy performance."""
        significance_tests = {}

        for strategy_name, performance in strategy_performances.items():
            tests = {}

            # Test if returns are significantly different from zero
            if len(performance.trades) > 10:
                trade_returns = [t.pnl_pct for t in performance.trades]

                # T-test against zero
                t_stat, p_value = stats.ttest_1samp(trade_returns, 0)
                tests['t_test_p_value'] = p_value
                tests['t_test_significant'] = p_value < 0.05

                # Test if Sharpe ratio is significantly positive
                if performance.volatility > 0:
                    # Bootstrap test for Sharpe ratio
                    bootstrap_sharpes = []
                    equity_returns = performance.equity_curve.pct_change().dropna()

                    for _ in range(1000):
                        sample_returns = np.random.choice(equity_returns, size=len(equity_returns), replace=True)
                        if np.std(sample_returns) > 0:
                            bootstrap_sharpe = np.mean(sample_returns) / np.std(sample_returns)
                            bootstrap_sharpes.append(bootstrap_sharpe)

                    if bootstrap_sharpes:
                        sharpe_p_value = np.mean(np.array(bootstrap_sharpes) <= 0)
                        tests['sharpe_bootstrap_p_value'] = sharpe_p_value
                        tests['sharpe_significant'] = sharpe_p_value < 0.05

            # Economic significance threshold
            tests['economically_significant'] = (
                performance.total_return > 0.05 and  # At least 5% return
                performance.sharpe_ratio > 0.5 and   # Reasonable risk-adjusted return
                performance.max_drawdown < 0.2       # Acceptable drawdown
            )

            significance_tests[strategy_name] = tests

        return significance_tests

    def _analyze_regime_performance(self,
                                  strategy_performances: Dict[str, StrategyPerformance],
                                  market_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze performance across different market regimes."""
        if not self.config.analyze_regime_performance:
            return {}

        regime_analysis = {}

        # Define regimes based on volatility
        if 'close' in market_data.columns:
            returns = market_data['close'].pct_change()
            volatility = returns.rolling(20).std()

            # Define regime thresholds
            vol_25 = volatility.quantile(0.33)
            vol_75 = volatility.quantile(0.67)

            # Classify regimes
            regimes = pd.Series(index=volatility.index, data='medium')
            regimes[volatility <= vol_25] = 'low_vol'
            regimes[volatility >= vol_75] = 'high_vol'

            # Analyze performance by regime
            for strategy_name, performance in strategy_performances.items():
                strategy_regime_analysis = {}

                # Get regime for each trade
                for regime_type in ['low_vol', 'medium', 'high_vol']:
                    regime_trades = []

                    for trade in performance.trades:
                        if trade.entry_time in regimes.index:
                            trade_regime = regimes.loc[trade.entry_time]
                            if trade_regime == regime_type:
                                regime_trades.append(trade)

                    if regime_trades:
                        regime_returns = [t.pnl_pct for t in regime_trades]
                        strategy_regime_analysis[f'{regime_type}_return'] = np.mean(regime_returns)
                        strategy_regime_analysis[f'{regime_type}_win_rate'] = np.mean([r > 0 for r in regime_returns])
                        strategy_regime_analysis[f'{regime_type}_trades'] = len(regime_trades)
                    else:
                        strategy_regime_analysis[f'{regime_type}_return'] = 0.0
                        strategy_regime_analysis[f'{regime_type}_win_rate'] = 0.0
                        strategy_regime_analysis[f'{regime_type}_trades'] = 0

                regime_analysis[strategy_name] = strategy_regime_analysis

        return regime_analysis

    def _perform_comparative_analysis(self,
                                    strategy_performances: Dict[str, StrategyPerformance]) -> Dict[str, Any]:
        """Perform comparative analysis of strategies."""
        comparative_analysis = {}

        if not strategy_performances:
            return comparative_analysis

        # Performance rankings
        sharpe_ranking = sorted(strategy_performances.items(),
                              key=lambda x: x[1].sharpe_ratio, reverse=True)
        return_ranking = sorted(strategy_performances.items(),
                              key=lambda x: x[1].total_return, reverse=True)
        drawdown_ranking = sorted(strategy_performances.items(),
                                key=lambda x: x[1].max_drawdown)

        comparative_analysis['best_sharpe'] = sharpe_ranking[0][0] if sharpe_ranking else None
        comparative_analysis['best_return'] = return_ranking[0][0] if return_ranking else None
        comparative_analysis['best_drawdown'] = drawdown_ranking[0][0] if drawdown_ranking else None

        # Performance statistics
        sharpe_ratios = [p.sharpe_ratio for p in strategy_performances.values()]
        returns = [p.total_return for p in strategy_performances.values()]
        drawdowns = [p.max_drawdown for p in strategy_performances.values()]

        comparative_analysis['sharpe_stats'] = {
            'mean': np.mean(sharpe_ratios),
            'std': np.std(sharpe_ratios),
            'min': np.min(sharpe_ratios),
            'max': np.max(sharpe_ratios)
        }

        comparative_analysis['return_stats'] = {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'max': np.max(returns)
        }

        comparative_analysis['drawdown_stats'] = {
            'mean': np.mean(drawdowns),
            'std': np.std(drawdowns),
            'min': np.min(drawdowns),
            'max': np.max(drawdowns)
        }

        # Strategy type analysis
        type_performance = {}
        for strategy_name, performance in strategy_performances.items():
            strategy_type = performance.strategy_type.value
            if strategy_type not in type_performance:
                type_performance[strategy_type] = []
            type_performance[strategy_type].append(performance.sharpe_ratio)

        comparative_analysis['type_performance'] = {
            k: {'mean': np.mean(v), 'count': len(v)}
            for k, v in type_performance.items()
        }

        return comparative_analysis

    def _generate_validation_summary(self,
                                   strategy_performances: Dict[str, StrategyPerformance]) -> Dict[str, Any]:
        """Generate validation summary."""
        summary = {}

        if not strategy_performances:
            summary['validation_result'] = 'FAILED'
            summary['reason'] = 'No successful strategy backtests'
            return summary

        # Overall validation assessment
        successful_strategies = [p for p in strategy_performances.values()
                               if p.total_trades >= self.config.min_trades]

        if not successful_strategies:
            summary['validation_result'] = 'FAILED'
            summary['reason'] = 'Insufficient trades in all strategies'
            return summary

        # Performance thresholds
        profitable_strategies = [p for p in successful_strategies if p.total_return > 0]
        high_sharpe_strategies = [p for p in successful_strategies if p.sharpe_ratio > 0.5]
        low_drawdown_strategies = [p for p in successful_strategies if p.max_drawdown < 0.15]

        # Validation scoring
        profitability_score = len(profitable_strategies) / len(successful_strategies)
        sharpe_score = len(high_sharpe_strategies) / len(successful_strategies)
        drawdown_score = len(low_drawdown_strategies) / len(successful_strategies)

        overall_score = (profitability_score + sharpe_score + drawdown_score) / 3

        # Determine validation result
        if overall_score >= 0.7:
            validation_result = 'EXCELLENT'
        elif overall_score >= 0.5:
            validation_result = 'GOOD'
        elif overall_score >= 0.3:
            validation_result = 'ACCEPTABLE'
        else:
            validation_result = 'POOR'

        summary['validation_result'] = validation_result
        summary['overall_score'] = overall_score
        summary['profitability_score'] = profitability_score
        summary['sharpe_score'] = sharpe_score
        summary['drawdown_score'] = drawdown_score
        summary['successful_strategies'] = len(successful_strategies)
        summary['total_strategies'] = len(strategy_performances)

        # Best strategy summary
        if successful_strategies:
            best_strategy = max(successful_strategies, key=lambda x: x.sharpe_ratio)
            summary['best_strategy'] = {
                'name': best_strategy.strategy_name,
                'total_return': best_strategy.total_return,
                'sharpe_ratio': best_strategy.sharpe_ratio,
                'max_drawdown': best_strategy.max_drawdown,
                'total_trades': best_strategy.total_trades
            }

        return summary

    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        if not self.validation_results:
            return "No validation results available. Run validate_through_backtesting() first."

        report_lines = [
            "# Backtesting-Integrated Labeling Validation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"**Validation Result**: {self.validation_results.validation_summary.get('validation_result', 'N/A')}",
            f"**Overall Score**: {self.validation_results.validation_summary.get('overall_score', 0):.3f}",
            f"**Successful Strategies**: {self.validation_results.validation_summary.get('successful_strategies', 0)}/{self.validation_results.validation_summary.get('total_strategies', 0)}",
            ""
        ]

        # Best strategy summary
        best_strategy = self.validation_results.validation_summary.get('best_strategy')
        if best_strategy:
            report_lines.extend([
                "## Best Performing Strategy",
                f"**Strategy**: {best_strategy['name']}",
                f"**Total Return**: {best_strategy['total_return']:.2%}",
                f"**Sharpe Ratio**: {best_strategy['sharpe_ratio']:.3f}",
                f"**Maximum Drawdown**: {best_strategy['max_drawdown']:.2%}",
                f"**Total Trades**: {best_strategy['total_trades']}",
                ""
            ])

        # Strategy performance summary
        report_lines.extend([
            "## Strategy Performance Summary",
            ""
        ])

        for strategy_name, performance in self.validation_results.strategy_performances.items():
            report_lines.extend([
                f"### {strategy_name}",
                f"- Total Return: {performance.total_return:.2%}",
                f"- Sharpe Ratio: {performance.sharpe_ratio:.3f}",
                f"- Max Drawdown: {performance.max_drawdown:.2%}",
                f"- Win Rate: {performance.win_rate:.2%}",
                f"- Total Trades: {performance.total_trades}",
                ""
            ])

        # Economic significance
        report_lines.extend([
            "## Economic Significance Tests",
            ""
        ])

        for strategy_name, tests in self.validation_results.economic_significance_tests.items():
            economically_sig = tests.get('economically_significant', False)
            status = "✅ Significant" if economically_sig else "⚠️ Not Significant"
            report_lines.extend([
                f"### {strategy_name}: {status}",
                f"- T-test p-value: {tests.get('t_test_p_value', 'N/A'):.4f}" if 't_test_p_value' in tests else "",
                f"- Sharpe bootstrap p-value: {tests.get('sharpe_bootstrap_p_value', 'N/A'):.4f}" if 'sharpe_bootstrap_p_value' in tests else "",
                ""
            ])

        # Comparative analysis
        comparative = self.validation_results.comparative_analysis
        if comparative:
            report_lines.extend([
                "## Comparative Analysis",
                f"**Best Sharpe Ratio**: {comparative.get('best_sharpe', 'N/A')}",
                f"**Best Total Return**: {comparative.get('best_return', 'N/A')}",
                f"**Best Drawdown**: {comparative.get('best_drawdown', 'N/A')}",
                "",
                "### Performance Statistics",
                f"- Average Sharpe Ratio: {comparative.get('sharpe_stats', {}).get('mean', 0):.3f}",
                f"- Average Total Return: {comparative.get('return_stats', {}).get('mean', 0):.2%}",
                f"- Average Max Drawdown: {comparative.get('drawdown_stats', {}).get('mean', 0):.2%}",
                ""
            ])

        return "\n".join(report_lines)

# Convenience functions
def validate_labels_through_backtesting(labeled_data: pd.DataFrame,
                                       market_data: pd.DataFrame,
                                       config: Optional[BacktestingConfig] = None) -> BacktestingValidationResult:
    """Convenience function for backtesting validation."""
    validator = BacktestingIntegratedValidator(config)
    return validator.validate_through_backtesting(labeled_data, market_data)

def generate_backtesting_validation_report(labeled_data: pd.DataFrame,
                                         market_data: pd.DataFrame,
                                         config: Optional[BacktestingConfig] = None) -> str:
    """Convenience function to generate backtesting validation report."""
    validator = BacktestingIntegratedValidator(config)
    validator.validate_through_backtesting(labeled_data, market_data)
    return validator.generate_validation_report()
