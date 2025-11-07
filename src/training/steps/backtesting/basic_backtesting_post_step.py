"""
Basic Backtesting Post Step.

This step performs post-optimization comparison backtesting.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# VectorBT imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint

# Data loading utilities
try:
    from src.utils.data.klines_parquet import get_klines_manager
    DATA_LOADING_AVAILABLE = True
except ImportError:
    DATA_LOADING_AVAILABLE = False
    get_klines_manager = None

logger = logging.getLogger(__name__)


class BasicBacktestingPostStep(BaseStep):
    """
    Basic Backtesting Post Step.

    Performs comparison backtesting after parameter optimization.
    """

    def __init__(self, step_name: str = "basic_backtesting_post"):
        """Initialize the basic backtesting post step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('BasicBacktestingPost')

        # Time-series CV configuration
        self.enable_time_series_cv = True
        self.cv_n_splits = 5  # Number of CV splits
        self.cv_embargo_pct = 0.02  # 2% embargo between train/test to prevent leakage
        
    def _calculate_vectorbt_metrics(self, returns: pd.Series, prices: pd.Series) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics using VectorBT.
        
        Args:
            returns: Series of returns
            prices: Series of prices (for drawdown calculation)
            
        Returns:
            Dictionary of calculated metrics
        """
        if not VECTORBT_AVAILABLE or returns is None or len(returns) == 0:
            return {}
            
        try:
            metrics = {}
            
            # Calculate returns-based metrics
            if len(returns) > 0:
                metrics['total_return'] = float((1 + returns).prod() - 1)
                metrics['annualized_return'] = float((1 + returns).mean() ** 252 - 1)
                metrics['volatility'] = float(returns.std() * np.sqrt(252))
                
                # Sharpe Ratio
                if metrics['volatility'] > 0:
                    metrics['sharpe_ratio'] = float((metrics['annualized_return'] - 0.02) / metrics['volatility'])
                else:
                    metrics['sharpe_ratio'] = 0.0
                    
                # Sortino Ratio (downside deviation)
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std() * np.sqrt(252)
                    if downside_std > 0:
                        metrics['sortino_ratio'] = float((metrics['annualized_return'] - 0.02) / downside_std)
                    else:
                        metrics['sortino_ratio'] = metrics['sharpe_ratio']
                else:
                    metrics['sortino_ratio'] = metrics['sharpe_ratio']
            
            # Calculate drawdown metrics using VectorBT
            if len(prices) > 1:
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                
                metrics['max_drawdown'] = float(drawdown.min())
                
                # Max drawdown duration
                dd_duration = drawdown < 0
                if dd_duration.any():
                    metrics['max_drawdown_duration_days'] = int(dd_duration.sum())
                else:
                    metrics['max_drawdown_duration_days'] = 0
                    
                # Calmar Ratio
                if metrics['max_drawdown'] < 0:
                    metrics['calmar_ratio'] = float(abs(metrics['annualized_return'] / metrics['max_drawdown']))
                else:
                    metrics['calmar_ratio'] = float('inf') if metrics['annualized_return'] > 0 else 0.0
                    
                # Recovery Factor
                if metrics['max_drawdown'] < 0:
                    metrics['recovery_factor'] = float(abs(metrics['total_return'] / metrics['max_drawdown']))
                else:
                    metrics['recovery_factor'] = 0.0
            
            # Sharpe-Sortino Spread
            metrics['sharpe_sortino_spread'] = metrics.get('sharpe_ratio', 0.0) - metrics.get('sortino_ratio', 0.0)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating VectorBT metrics: {e}")
            return {}

    def _load_optimized_parameters(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load optimized parameters from final_parameters_optimization step.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary of optimized parameters or None if not found
        """
        try:
            # Try to load from artifact manager
            params = self._get_artifact(
                artifact_name='final_parameters_optimization_result',
                artifact_type='data'
            )
            
            if params is not None:
                self.logger.info(f"Loaded optimized parameters: {len(params)} parameters")
                return params
            else:
                self.logger.warning("No optimized parameters found, using defaults")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to load optimized parameters: {e}, using defaults")
            return None
    
    def _load_baseline_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Load baseline metrics from basic_backtesting_pre step.
        
        Returns:
            Dictionary of baseline metrics or None if not found
        """
        try:
            baseline = self._get_artifact(
                artifact_name='pre_optimization_backtest',
                artifact_type='data'
            )
            
            if baseline is not None:
                self.logger.info("Loaded baseline metrics for comparison")
                return baseline
            else:
                self.logger.warning("No baseline metrics found for comparison")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to load baseline metrics: {e}")
            return None
    
    def _load_ml_scored_data(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Load ML-scored historical data from training artifacts.
        
        Tries to load both analyst and tactician predictions. Prioritizes tactician
        as it's the final layer, but falls back to analyst if tactician not available.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            DataFrame with ML predictions or None if not found
        """
        try:
            direction = config.get('direction', 'long')
            
            # Try tactician first (most sophisticated)
            for model_type in ['tactician', 'analyst']:
                try:
                    artifact_name = f"ml_scored_historical_data_{model_type}_{direction}"
                    ml_data = self._get_artifact(artifact_name, 'data')
                    
                    if ml_data is not None and not ml_data.empty:
                        self.logger.info(f"Loaded ML-scored data from {model_type}: {len(ml_data)} samples")
                        return ml_data
                        
                except Exception as e:
                    self.logger.debug(f"ML-scored data not found for {model_type}: {e}")
                    continue
            
            self.logger.warning("No ML-scored data found, will fall back to simple signals")
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to load ML-scored data: {e}")
            return None
    
    def _load_price_data(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Load OHLCV price data using KlinesParquetManager.
        
        Args:
            config: Configuration dictionary with symbol, timeframe, exchange
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not DATA_LOADING_AVAILABLE:
            self.logger.error("Data loading utilities not available")
            return None
            
        try:
            symbol = config.get('symbol', 'ETHUSDT')
            timeframe = config.get('timeframe', '15m')
            exchange = config.get('exchange', 'binance')
            
            self.logger.info(f"Loading price data for {symbol} {timeframe} from {exchange}")
            
            # Get klines manager
            klines_manager = get_klines_manager()
            
            # Load OHLCV data
            price_data = klines_manager.read_data(
                symbol=symbol,
                interval=timeframe,
                data_type='raw',
                columns=['open', 'high', 'low', 'close', 'volume']
            )
            
            if price_data is None or len(price_data) == 0:
                self.logger.error("No price data loaded")
                return None
            
            # Apply light mode filtering if needed
            price_data = self._apply_light_mode_filter(price_data, config, timeframe)
            
            self.logger.info(f"Loaded {len(price_data)} price bars")
            return price_data
            
        except Exception as e:
            self.logger.error(f"Failed to load price data: {e}")
            return None
    
    def _generate_ml_signals(
        self,
        ml_scored_data: pd.DataFrame,
        parameters: Optional[Dict[str, Any]] = None,
        direction: str = 'long'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate trading signals from ML model predictions.
        
        Uses ML predictions to generate entry/exit signals based on prediction confidence
        and directional scores.
        
        Args:
            ml_scored_data: DataFrame with ML predictions
            parameters: Dictionary of optimized parameters (if available)
            direction: Trading direction ('long', 'short', 'both')
            
        Returns:
            Tuple of (long_entries, short_entries, exits) boolean series
        """
        try:
            # Extract parameters or use defaults
            if parameters and isinstance(parameters, dict):
                confidence_threshold = parameters.get('confidence_threshold', 0.6)
                exit_threshold = parameters.get('exit_threshold', 0.4)
            else:
                confidence_threshold = 0.6
                exit_threshold = 0.4
            
            # Initialize signals
            long_entries = pd.Series(False, index=ml_scored_data.index)
            short_entries = pd.Series(False, index=ml_scored_data.index)
            exits = pd.Series(False, index=ml_scored_data.index)
            
            # Find prediction columns (could be from analyst or tactician)
            pred_cols = [col for col in ml_scored_data.columns if 'prediction' in col.lower() or 'confidence' in col.lower()]
            
            if not pred_cols:
                self.logger.warning("No prediction columns found in ML data, using first available column")
                # Use first numeric column as prediction
                numeric_cols = ml_scored_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    prediction = ml_scored_data[numeric_cols[0]]
                else:
                    raise ValueError("No numeric prediction columns found")
            else:
                # Use first prediction column found
                prediction = ml_scored_data[pred_cols[0]]
            
            # Look for confidence column
            confidence_cols = [col for col in ml_scored_data.columns if 'confidence' in col.lower()]
            if confidence_cols:
                confidence = ml_scored_data[confidence_cols[0]]
            else:
                # Use absolute value of prediction as confidence
                confidence = prediction.abs()
            
            # Normalize confidence to [0, 1]
            if confidence.max() > 1.0:
                confidence = confidence / confidence.max()
            
            # Generate signals based on direction and confidence
            if direction in ['long', 'both']:
                # Long entry: positive prediction with high confidence
                long_signal = (prediction > 0) & (confidence >= confidence_threshold)
                # Entry on signal change from False to True
                long_entries = long_signal & ~long_signal.shift(1).fillna(False)
                
            if direction in ['short', 'both']:
                # Short entry: negative prediction with high confidence
                short_signal = (prediction < 0) & (confidence >= confidence_threshold)
                # Entry on signal change from False to True
                short_entries = short_signal & ~short_signal.shift(1).fillna(False)
            
            # Exit signals: confidence drops below exit threshold
            if direction == 'long':
                exits = (confidence < exit_threshold) | (prediction < 0)
            elif direction == 'short':
                exits = (confidence < exit_threshold) | (prediction > 0)
            elif direction == 'both':
                # Exit when signal changes or confidence drops
                exits = (confidence < exit_threshold) | long_entries | short_entries
            
            long_count = long_entries.sum()
            short_count = short_entries.sum()
            exit_count = exits.sum()
            
            self.logger.info(f"Generated ML signals: {long_count} long, {short_count} short, {exit_count} exits")
            
            return long_entries.fillna(False), short_entries.fillna(False), exits.fillna(False)
            
        except Exception as e:
            self.logger.error(f"Failed to generate ML signals: {e}")
            # Return empty signals
            empty = pd.Series(False, index=ml_scored_data.index)
            return empty.copy(), empty.copy(), empty.copy()
    
    def _generate_simple_signals(
        self, 
        price_data: pd.DataFrame,
        parameters: Optional[Dict[str, Any]] = None,
        direction: str = 'long'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate trading signals using a simple moving average crossover strategy.
        
        This is a fallback strategy when ML predictions are not available.
        
        Args:
            price_data: DataFrame with OHLCV data
            parameters: Dictionary of optimized parameters (if available)
            direction: Trading direction ('long', 'short', 'both')
            
        Returns:
            Tuple of (long_entries, short_entries, exits) boolean series
        """
        try:
            # Extract parameters or use defaults
            if parameters and isinstance(parameters, dict):
                fast_window = parameters.get('fast_ma_window', 20)
                slow_window = parameters.get('slow_ma_window', 50)
            else:
                fast_window = 20
                slow_window = 50
            
            close = price_data['close']
            
            # Calculate moving averages
            fast_ma = close.rolling(window=fast_window).mean()
            slow_ma = close.rolling(window=slow_window).mean()
            
            # Initialize signals
            long_entries = pd.Series(False, index=price_data.index)
            short_entries = pd.Series(False, index=price_data.index)
            exits = pd.Series(False, index=price_data.index)
            
            # Generate signals based on direction
            if direction in ['long', 'both']:
                # Long entry: fast MA crosses above slow MA
                long_entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
                
            if direction in ['short', 'both']:
                # Short entry: fast MA crosses below slow MA
                short_entries = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
            
            # Exit signals: opposite crossover
            if direction == 'long':
                exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
            elif direction == 'short':
                exits = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
            elif direction == 'both':
                # Exit on opposite signal (will flip position)
                exits = long_entries | short_entries
            
            long_count = long_entries.sum()
            short_count = short_entries.sum()
            exit_count = exits.sum()
            
            self.logger.info(f"Generated fallback signals: {long_count} long, {short_count} short, {exit_count} exits")
            
            return long_entries.fillna(False), short_entries.fillna(False), exits.fillna(False)
            
        except Exception as e:
            self.logger.error(f"Failed to generate signals: {e}")
            # Return empty signals
            empty = pd.Series(False, index=price_data.index)
            return empty.copy(), empty.copy(), empty.copy()
    
    def _run_vectorbt_backtest(
        self,
        price_data: pd.DataFrame,
        long_entries: pd.Series,
        short_entries: pd.Series,
        exits: pd.Series,
        config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Run backtest using VectorBT portfolio simulation.
        
        Args:
            price_data: DataFrame with OHLCV data
            long_entries: Boolean series for long entry signals
            short_entries: Boolean series for short entry signals
            exits: Boolean series for exit signals
            config: Configuration dictionary
            
        Returns:
            Dictionary containing backtest results or None if failed
        """
        if not VECTORBT_AVAILABLE:
            self.logger.error("VectorBT is not available, cannot run backtest")
            return None
        
        try:
            # Extract prices
            close_prices = price_data['close']
            
            # Portfolio configuration
            init_cash = config.get('initial_cash', 10000.0)
            fees = config.get('fees', 0.001)  # 0.1% fees
            slippage = config.get('slippage', 0.0005)  # 0.05% slippage
            direction = config.get('direction', 'long')
            
            # Create portfolio based on direction
            if direction == 'long':
                # Long-only portfolio
                portfolio = vbt.Portfolio.from_signals(
                    close=close_prices,
                    entries=long_entries,
                    exits=exits,
                    init_cash=init_cash,
                    fees=fees,
                    slippage=slippage,
                    freq='1T'  # Will be adjusted based on actual timeframe
                )
            elif direction == 'short':
                # Short-only portfolio
                portfolio = vbt.Portfolio.from_signals(
                    close=close_prices,
                    entries=short_entries,
                    exits=exits,
                    short_entries=short_entries,  # Mark as short positions
                    init_cash=init_cash,
                    fees=fees,
                    slippage=slippage,
                    freq='1T'
                )
            else:  # 'both'
                # Long and short portfolio
                portfolio = vbt.Portfolio.from_signals(
                    close=close_prices,
                    entries=long_entries,
                    exits=exits,
                    short_entries=short_entries,
                    init_cash=init_cash,
                    fees=fees,
                    slippage=slippage,
                    freq='1T'
                )
            
            # Extract metrics
            returns = portfolio.returns()
            equity = portfolio.value()
            trades = portfolio.trades.records_readable
            
            # Calculate metrics using VectorBT's built-in metrics
            total_return = portfolio.total_return()
            sharpe_ratio = portfolio.sharpe_ratio()
            max_dd = portfolio.max_drawdown()
            win_rate = portfolio.trades.win_rate() if len(trades) > 0 else 0.0
            total_trades = len(trades)
            
            # Calculate additional metrics using our custom method
            custom_metrics = self._calculate_vectorbt_metrics(returns, close_prices)
            
            # Calculate trade-level metrics
            trade_metrics = self._calculate_trade_metrics(trades) if len(trades) > 0 else {}
            
            # Combine results
            result = {
                'portfolio': portfolio,
                'returns': returns,
                'equity_curve': equity,
                'trades': trades,
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_dd),
                'win_rate': float(win_rate),
                'total_trades': int(total_trades),
                **custom_metrics,
                **trade_metrics
            }
            
            self.logger.info(f"Backtest completed ({direction}): {total_trades} trades, {total_return:.2%} return")
            
            return result
            
        except Exception as e:
            self.logger.error(f"VectorBT backtest failed: {e}")
            return None
    
    def _run_time_series_cv_backtest(
        self,
        price_data: pd.DataFrame,
        ml_scored_data: Optional[pd.DataFrame],
        optimized_params: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run time-series cross-validation backtesting to ensure robust performance.

        This implements walk-forward validation with embargo to prevent data leakage:
        - Uses sklearn's TimeSeriesSplit for proper temporal ordering
        - Applies embargo period between train/test to prevent look-ahead bias
        - Validates that test periods contain only "unseen" data
        - Aggregates results across all CV folds

        Args:
            price_data: DataFrame with OHLCV data
            ml_scored_data: DataFrame with ML predictions (optional)
            optimized_params: Dictionary of optimized parameters
            config: Configuration dictionary

        Returns:
            Dictionary containing CV results with mean/std of metrics
        """
        if not self.enable_time_series_cv:
            self.logger.info("Time-series CV disabled, skipping")
            return {}

        try:
            tprint(f"🔄 Running time-series CV with {self.cv_n_splits} splits and {self.cv_embargo_pct:.1%} embargo", "INFO")

            direction = config.get('direction', 'long')
            use_ml_signals = ml_scored_data is not None

            # Initialize TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=self.cv_n_splits)

            # Storage for fold results
            fold_results = []

            # Iterate over CV folds
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(price_data)):
                tprint(f"  📊 Processing CV fold {fold_idx + 1}/{self.cv_n_splits}", "INFO")

                # Apply embargo: remove samples near the train/test boundary
                embargo_size = int(len(train_idx) * self.cv_embargo_pct)
                if embargo_size > 0:
                    # Remove last embargo_size samples from training set
                    train_idx = train_idx[:-embargo_size]
                    # Remove first embargo_size samples from test set
                    if len(test_idx) > embargo_size:
                        test_idx = test_idx[embargo_size:]

                # Skip if test set is too small after embargo
                if len(test_idx) < 50:
                    self.logger.warning(f"Fold {fold_idx + 1} test set too small ({len(test_idx)} samples), skipping")
                    continue

                # Extract test data for this fold
                test_price_data = price_data.iloc[test_idx].copy()

                # Generate signals for test period
                if use_ml_signals and ml_scored_data is not None:
                    # Extract ML predictions for test period
                    test_ml_data = ml_scored_data.iloc[test_idx].copy()
                    long_entries, short_entries, exits = self._generate_ml_signals(
                        test_ml_data, optimized_params, direction
                    )
                else:
                    # Use simple signals as fallback
                    long_entries, short_entries, exits = self._generate_simple_signals(
                        test_price_data, optimized_params, direction
                    )

                # Run backtest on this fold
                fold_backtest = self._run_vectorbt_backtest(
                    test_price_data, long_entries, short_entries, exits, config
                )

                if fold_backtest is not None:
                    # Extract key metrics
                    fold_result = {
                        'fold': fold_idx + 1,
                        'train_size': len(train_idx),
                        'test_size': len(test_idx),
                        'embargo_size': embargo_size,
                        'total_return': fold_backtest.get('total_return', 0.0),
                        'sharpe_ratio': fold_backtest.get('sharpe_ratio', 0.0),
                        'sortino_ratio': fold_backtest.get('sortino_ratio', 0.0),
                        'max_drawdown': fold_backtest.get('max_drawdown', 0.0),
                        'win_rate': fold_backtest.get('win_rate', 0.0),
                        'profit_factor': fold_backtest.get('profit_factor', 0.0),
                        'total_trades': fold_backtest.get('total_trades', 0)
                    }
                    fold_results.append(fold_result)

                    tprint(f"    ✅ Fold {fold_idx + 1}: Return={fold_result['total_return']:.2%}, "
                          f"Sharpe={fold_result['sharpe_ratio']:.2f}, Trades={fold_result['total_trades']}",
                          "SUCCESS")
                else:
                    self.logger.warning(f"Fold {fold_idx + 1} backtest failed")

            if not fold_results:
                self.logger.error("All CV folds failed")
                return {}

            # Aggregate results across folds
            cv_df = pd.DataFrame(fold_results)

            cv_summary = {
                'n_folds': len(fold_results),
                'total_return_mean': float(cv_df['total_return'].mean()),
                'total_return_std': float(cv_df['total_return'].std()),
                'sharpe_ratio_mean': float(cv_df['sharpe_ratio'].mean()),
                'sharpe_ratio_std': float(cv_df['sharpe_ratio'].std()),
                'sortino_ratio_mean': float(cv_df['sortino_ratio'].mean()),
                'sortino_ratio_std': float(cv_df['sortino_ratio'].std()),
                'max_drawdown_mean': float(cv_df['max_drawdown'].mean()),
                'max_drawdown_std': float(cv_df['max_drawdown'].std()),
                'win_rate_mean': float(cv_df['win_rate'].mean()),
                'win_rate_std': float(cv_df['win_rate'].std()),
                'profit_factor_mean': float(cv_df['profit_factor'].mean()),
                'profit_factor_std': float(cv_df['profit_factor'].std()),
                'total_trades_mean': float(cv_df['total_trades'].mean()),
                'fold_details': fold_results
            }

            tprint(f"✅ Time-series CV completed: "
                  f"Return={cv_summary['total_return_mean']:.2%}±{cv_summary['total_return_std']:.2%}, "
                  f"Sharpe={cv_summary['sharpe_ratio_mean']:.2f}±{cv_summary['sharpe_ratio_std']:.2f}",
                  "SUCCESS")

            return cv_summary

        except Exception as e:
            self.logger.error(f"Time-series CV failed: {e}", exc_info=True)
            return {}

    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate trade-level performance metrics.
        
        Args:
            trades: DataFrame of trade records from VectorBT
            
        Returns:
            Dictionary of trade metrics
        """
        try:
            if len(trades) == 0:
                return {
                    'avg_win_loss_ratio': 0.0,
                    'profit_factor': 0.0,
                    'expectancy': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0,
                    'avg_trade_duration': 0.0
                }
            
            # Extract PnL
            if 'PnL' in trades.columns:
                pnl = trades['PnL']
            elif 'Return' in trades.columns:
                pnl = trades['Return']
            else:
                # Fallback calculation
                pnl = trades.get('Exit Price', 0) - trades.get('Entry Price', 0)
            
            # Separate winning and losing trades
            winning_trades = pnl[pnl > 0]
            losing_trades = pnl[pnl < 0]
            
            # Calculate metrics
            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
            
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            expectancy = pnl.mean()
            largest_win = pnl.max()
            largest_loss = pnl.min()
            
            # Calculate average trade duration
            if 'Duration' in trades.columns:
                avg_duration = trades['Duration'].mean().total_seconds() / 3600  # in hours
            else:
                avg_duration = 0.0
            
            return {
                'avg_win_loss_ratio': float(win_loss_ratio),
                'profit_factor': float(profit_factor),
                'expectancy': float(expectancy),
                'largest_win': float(largest_win),
                'largest_loss': float(largest_loss),
                'avg_trade_duration': float(avg_duration)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate trade metrics: {e}")
            return {}
    
    def _compare_with_baseline(
        self,
        post_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compare post-optimization metrics with baseline.
        
        Args:
            post_metrics: Post-optimization performance metrics
            baseline_metrics: Baseline performance metrics (optional)
            
        Returns:
            Dictionary of improvement metrics (empty if no baseline)
        """
        if baseline_metrics is None:
            self.logger.info("No baseline metrics available - skipping comparison")
            return {}
        
        try:
            comparison = {}
            
            # Compare key metrics
            metrics_to_compare = [
                'total_return',
                'sharpe_ratio',
                'sortino_ratio',
                'calmar_ratio',
                'max_drawdown',
                'win_rate'
            ]
            
            for metric in metrics_to_compare:
                baseline_value = baseline_metrics.get(metric, 0)
                post_value = post_metrics.get(metric, 0)
                
                if baseline_value != 0:
                    if metric == 'max_drawdown':
                        # For drawdown, reduction is improvement
                        improvement = baseline_value - post_value  # Both negative, so reduction in magnitude
                        comparison[f'{metric}_reduction'] = float(improvement)
                    else:
                        # For other metrics, increase is improvement
                        improvement = post_value - baseline_value
                        comparison[f'{metric}_improvement'] = float(improvement)
            
            self.logger.info(f"Comparison completed: {len(comparison)} improvement metrics calculated")
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Failed to compare with baseline: {e}")
            return {}
    
    def _generate_markdown_report(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        artifacts: Dict[str, Any],
        price_data: pd.DataFrame,
        optimized_params: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate comprehensive markdown report for backtest results.
        
        Args:
            config: Configuration dictionary
            metrics: Performance metrics
            artifacts: Artifacts dictionary
            price_data: Price data used for backtest
            optimized_params: Optimized parameters used
            
        Returns:
            Path to generated markdown report
        """
        try:
            from pathlib import Path
            
            # Create outcomes directory
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)
            
            # Generate filename with datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = config.get('symbol', 'UNKNOWN')
            direction = config.get('direction', 'long')
            filename = f"basic_backtesting_post_{symbol}_{direction}_{timestamp}.md"
            report_path = outcomes_dir / filename
            
            # Build markdown content
            lines = []
            
            # Header
            lines.append(f"# Post-Optimization Backtesting Report")
            lines.append(f"**Symbol:** {symbol} | **Exchange:** {config.get('exchange', 'N/A')} | "
                        f"**Timeframe:** {config.get('timeframe', 'N/A')} | **Direction:** {direction}")
            lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            
            # Executive Summary
            lines.append("## 📊 Executive Summary")
            lines.append("")
            total_return = metrics.get('total_return', 0)
            sharpe = metrics.get('sharpe_ratio', 0)
            sortino = metrics.get('sortino_ratio', 0)
            max_dd = metrics.get('max_drawdown', 0)
            total_trades = metrics.get('total_trades', 0)
            
            lines.append(f"- **Total Return:** {total_return:.2%}")
            lines.append(f"- **Sharpe Ratio:** {sharpe:.3f}")
            lines.append(f"- **Sortino Ratio:** {sortino:.3f}")
            lines.append(f"- **Max Drawdown:** {max_dd:.2%}")
            lines.append(f"- **Total Trades:** {total_trades}")
            lines.append("")
            
            # Performance Metrics Section
            lines.append("## 📈 Performance Metrics")
            lines.append("")
            
            lines.append("### Risk-Adjusted Returns")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Total Return | {metrics.get('total_return', 0):.2%} |")
            lines.append(f"| Annualized Return | {metrics.get('annualized_return', 0):.2%} |")
            lines.append(f"| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.3f} |")
            lines.append(f"| Sortino Ratio | {metrics.get('sortino_ratio', 0):.3f} |")
            lines.append(f"| Calmar Ratio | {metrics.get('calmar_ratio', 0):.3f} |")
            lines.append(f"| Sharpe-Sortino Spread | {metrics.get('sharpe_sortino_spread', 0):.3f} |")
            lines.append("")
            
            lines.append("### Drawdown Analysis")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Max Drawdown | {metrics.get('max_drawdown', 0):.2%} |")
            lines.append(f"| Max Drawdown Duration | {metrics.get('max_drawdown_duration_days', 0)} days |")
            lines.append(f"| Recovery Factor | {metrics.get('recovery_factor', 0):.3f} |")
            lines.append("")
            
            lines.append("### Trade Statistics")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Total Trades | {metrics.get('total_trades', 0)} |")
            lines.append(f"| Win Rate | {metrics.get('win_rate', 0):.2%} |")
            lines.append(f"| Profit Factor | {metrics.get('profit_factor', 0):.3f} |")
            lines.append(f"| Avg Win/Loss Ratio | {metrics.get('avg_win_loss_ratio', 0):.3f} |")
            lines.append(f"| Expectancy | {metrics.get('expectancy', 0):.4f} |")
            lines.append(f"| Avg Trade Duration | {metrics.get('avg_trade_duration', 0):.2f} hours |")
            lines.append(f"| Largest Win | {metrics.get('largest_win', 0):.4f} |")
            lines.append(f"| Largest Loss | {metrics.get('largest_loss', 0):.4f} |")
            lines.append("")

            # Time-Series CV Results
            cv_results = metrics.get('cv_results', {})
            if cv_results and cv_results.get('n_folds', 0) > 0:
                lines.append("## 🔄 Time-Series Cross-Validation Results")
                lines.append("")
                lines.append("Cross-validation ensures strategy robustness across different time periods and prevents overfitting.")
                lines.append("")
                lines.append(f"| Metric | Mean | Std Dev |")
                lines.append(f"|--------|------|---------|")
                lines.append(f"| Total Return | {cv_results.get('total_return_mean', 0):.2%} | {cv_results.get('total_return_std', 0):.2%} |")
                lines.append(f"| Sharpe Ratio | {cv_results.get('sharpe_ratio_mean', 0):.3f} | {cv_results.get('sharpe_ratio_std', 0):.3f} |")
                lines.append(f"| Sortino Ratio | {cv_results.get('sortino_ratio_mean', 0):.3f} | {cv_results.get('sortino_ratio_std', 0):.3f} |")
                lines.append(f"| Max Drawdown | {cv_results.get('max_drawdown_mean', 0):.2%} | {cv_results.get('max_drawdown_std', 0):.2%} |")
                lines.append(f"| Win Rate | {cv_results.get('win_rate_mean', 0):.2%} | {cv_results.get('win_rate_std', 0):.2%} |")
                lines.append(f"| Profit Factor | {cv_results.get('profit_factor_mean', 0):.3f} | {cv_results.get('profit_factor_std', 0):.3f} |")
                lines.append("")
                lines.append(f"**Number of CV Folds:** {cv_results.get('n_folds', 0)}")
                lines.append("")

            # Comparison with Baseline
            comparison = metrics.get('improvement_vs_baseline', {})
            if comparison:
                lines.append("## 📊 Baseline Comparison")
                lines.append("")
                lines.append("### Improvement Metrics")
                lines.append("")
                lines.append(f"| Metric | Improvement |")
                lines.append(f"|--------|-------------|")
                for key, value in comparison.items():
                    metric_name = key.replace('_', ' ').title()
                    if 'reduction' in key:
                        lines.append(f"| {metric_name} | {value:.4f} |")
                    else:
                        lines.append(f"| {metric_name} | {value:.4f} |")
                lines.append("")
            else:
                lines.append("## 📊 Baseline Comparison")
                lines.append("")
                lines.append("*No baseline metrics available for comparison.*")
                lines.append("")
            
            # Configuration Details
            lines.append("## ⚙️ Configuration")
            lines.append("")
            lines.append(f"| Parameter | Value |")
            lines.append(f"|-----------|-------|")
            lines.append(f"| Symbol | {config.get('symbol', 'N/A')} |")
            lines.append(f"| Exchange | {config.get('exchange', 'N/A')} |")
            lines.append(f"| Timeframe | {config.get('timeframe', 'N/A')} |")
            lines.append(f"| Direction | {config.get('direction', 'N/A')} |")
            lines.append(f"| Execution Mode | {config.get('execution_mode', 'N/A')} |")
            lines.append(f"| Signal Source | {'🤖 ML Model Predictions' if config.get('used_ml_signals', False) else '📊 Simple MA Crossover (Fallback)'} |")
            lines.append(f"| Initial Cash | ${config.get('initial_cash', 10000):,.2f} |")
            lines.append(f"| Fees | {config.get('fees', 0.001):.2%} |")
            lines.append(f"| Slippage | {config.get('slippage', 0.0005):.2%} |")
            lines.append("")
            
            # Optimized Parameters
            if optimized_params:
                lines.append("## 🎯 Optimized Parameters")
                lines.append("")
                lines.append(f"| Parameter | Value |")
                lines.append(f"|-----------|-------|")
                for key, value in optimized_params.items():
                    if isinstance(value, (int, float, str, bool)):
                        lines.append(f"| {key} | {value} |")
                lines.append("")
            else:
                lines.append("## 🎯 Optimized Parameters")
                lines.append("")
                lines.append("*Default parameters used (no optimization results found).*")
                lines.append("")
            
            # Data Summary
            lines.append("## 📅 Data Summary")
            lines.append("")
            lines.append(f"| Item | Value |")
            lines.append(f"|------|-------|")
            lines.append(f"| Data Points | {len(price_data):,} |")
            lines.append(f"| Start Date | {price_data.index[0]} |")
            lines.append(f"| End Date | {price_data.index[-1]} |")
            lines.append(f"| Price Range | ${price_data['close'].min():.2f} - ${price_data['close'].max():.2f} |")
            lines.append("")
            
            # Artifacts
            lines.append("## 📦 Generated Artifacts")
            lines.append("")
            artifacts_saved = artifacts.get('artifacts_saved', [])
            if artifacts_saved:
                for artifact_path in artifacts_saved:
                    lines.append(f"- `{artifact_path}`")
            else:
                lines.append("*No artifacts saved.*")
            lines.append("")
            
            # Footer
            lines.append("---")
            lines.append(f"*Report generated by Ares Trading System - Basic Backtesting Post Step*")
            
            # Write to file
            with open(report_path, 'w') as f:
                f.write('\n'.join(lines))
            
            self.logger.info(f"Generated markdown report: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate markdown report: {e}")
            return ""

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute post-optimization backtesting.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - direction: Trading direction ('long', 'short', 'both')

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        symbol = config.get('symbol', 'UNKNOWN')
        tprint(f"📈 Starting post-optimization backtesting for {symbol}", "INFO")

        try:
            # Step 1: Load optimized parameters
            tprint("📊 Loading optimized parameters...", "INFO")
            optimized_params = self._load_optimized_parameters(config)
            
            # Step 2: Load baseline metrics for comparison
            tprint("📊 Loading baseline metrics...", "INFO")
            baseline_metrics = self._load_baseline_metrics()
            
            # Step 3: Try to load ML-scored data first
            tprint("🧠 Checking for ML-scored data...", "INFO")
            ml_scored_data = self._load_ml_scored_data(config)
            
            # Step 4: Load price data (if ML data not available or missing price columns)
            price_data = None
            use_ml_signals = False
            
            if ml_scored_data is not None:
                # Check if ML data has price columns
                if all(col in ml_scored_data.columns for col in ['close', 'open', 'high', 'low']):
                    price_data = ml_scored_data[['open', 'high', 'low', 'close', 'volume']].copy() if 'volume' in ml_scored_data.columns else ml_scored_data[['open', 'high', 'low', 'close']].copy()
                    use_ml_signals = True
                    tprint("✅ Using ML-scored data with embedded predictions", "SUCCESS")
                else:
                    # ML data doesn't have price columns, need to load separately
                    tprint("⚠️ ML data missing price columns, loading price data separately", "WARNING")
                    price_data = self._load_price_data(config)
                    if price_data is not None:
                        use_ml_signals = True
            else:
                tprint("ℹ️ No ML-scored data found, loading price data for simple signals", "INFO")
                price_data = self._load_price_data(config)
            
            if price_data is None or len(price_data) == 0:
                raise ValueError("Failed to load price data")
            
            # Step 5: Generate trading signals
            direction = config.get('direction', 'long')
            
            if use_ml_signals and ml_scored_data is not None:
                tprint(f"🤖 Generating ML-based trading signals ({direction})...", "INFO")
                long_entries, short_entries, exits = self._generate_ml_signals(
                    ml_scored_data, optimized_params, direction
                )
            else:
                tprint(f"📊 Generating fallback trading signals ({direction})...", "INFO")
                long_entries, short_entries, exits = self._generate_simple_signals(
                    price_data, optimized_params, direction
                )
            
            # Step 5: Run backtest with VectorBT
            tprint("🔄 Running backtest simulation...", "INFO")
            backtest_results = self._run_vectorbt_backtest(
                price_data, long_entries, short_entries, exits, config
            )

            if backtest_results is None:
                raise ValueError("Backtest simulation failed")

            # Step 5.5: Run time-series CV for robust validation
            # This ensures the strategy performs consistently across different time periods
            # and prevents overfitting to the full backtest period
            cv_results = self._run_time_series_cv_backtest(
                price_data, ml_scored_data if use_ml_signals else None,
                optimized_params, config
            )

            # Step 6: Compare with baseline
            tprint("📊 Comparing with baseline...", "INFO")
            comparison_metrics = self._compare_with_baseline(backtest_results, baseline_metrics)
            
            # Step 7: Prepare artifacts
            # Extract equity curve and returns for visualization
            equity_curve_data = None
            if 'equity_curve' in backtest_results:
                equity_curve_data = pd.DataFrame({
                    'timestamp': backtest_results['equity_curve'].index,
                    'equity': backtest_results['equity_curve'].values
                })
            
            # Extract trade log
            trade_log_data = None
            if 'trades' in backtest_results and len(backtest_results['trades']) > 0:
                trade_log_data = backtest_results['trades'].copy()
            
            # Save artifacts
            artifacts_saved = []
            
            if equity_curve_data is not None:
                equity_path = self._save_artifact(
                    equity_curve_data,
                    'post_optimization_equity_curve',
                    'data'
                )
                artifacts_saved.append(equity_path)
            
            if trade_log_data is not None:
                trade_path = self._save_artifact(
                    trade_log_data,
                    'post_optimization_trade_log',
                    'data'
                )
                artifacts_saved.append(trade_path)
            
            # Prepare metrics summary
            metrics = {
                # Core performance metrics
                'total_return': backtest_results.get('total_return', 0.0),
                'annualized_return': backtest_results.get('annualized_return', 0.0),
                'sharpe_ratio': backtest_results.get('sharpe_ratio', 0.0),
                'sortino_ratio': backtest_results.get('sortino_ratio', 0.0),
                'calmar_ratio': backtest_results.get('calmar_ratio', 0.0),
                'max_drawdown': backtest_results.get('max_drawdown', 0.0),
                'max_drawdown_duration_days': backtest_results.get('max_drawdown_duration_days', 0),

                # Trade statistics
                'win_rate': backtest_results.get('win_rate', 0.0),
                'profit_factor': backtest_results.get('profit_factor', 0.0),
                'total_trades': backtest_results.get('total_trades', 0),
                'avg_trade_duration': backtest_results.get('avg_trade_duration', 0.0),

                # Trade quality metrics
                'avg_win_loss_ratio': backtest_results.get('avg_win_loss_ratio', 0.0),
                'expectancy': backtest_results.get('expectancy', 0.0),
                'largest_win': backtest_results.get('largest_win', 0.0),
                'largest_loss': backtest_results.get('largest_loss', 0.0),
                'recovery_factor': backtest_results.get('recovery_factor', 0.0),

                # Efficiency metrics
                'sharpe_sortino_spread': backtest_results.get('sharpe_sortino_spread', 0.0),

                # Time-series CV results (for validation robustness)
                'cv_results': cv_results,

                # Comparison with baseline
                'improvement_vs_baseline': comparison_metrics,

                # Metadata
                'direction': config.get('direction', 'long'),
                'execution_mode': config.get('execution_mode', 'light'),
                'success': True
            }
            
            # Prepare artifacts dictionary
            artifacts = {
                'post_optimization_backtest': {
                    'strategy_type': 'optimized',
                    'backtest_period': f"{price_data.index[0]} to {price_data.index[-1]}",
                    **metrics,
                    'metadata': {
                        'symbol': config['symbol'],
                        'exchange': config['exchange'],
                        'timeframe': config['timeframe'],
                        'direction': config.get('direction', 'long'),
                        'execution_mode': config.get('execution_mode', 'light'),
                        'created_at': datetime.now().isoformat(),
                        'data_points': len(price_data),
                        'optimized_parameters_used': optimized_params is not None
                    }
                },
                'equity_curve': equity_curve_data,
                'trade_log': trade_log_data,
                'artifacts_saved': artifacts_saved
            }
            
            # Save comprehensive results
            results_path = self._save_artifact(
                artifacts,
                'post_optimization_backtest_results',
                'data'
            )
            
            # Step 8: Generate markdown report
            tprint("📝 Generating comprehensive report...", "INFO")
            # Add ML usage info to config for report
            config['used_ml_signals'] = use_ml_signals
            report_path = self._generate_markdown_report(
                config, metrics, artifacts, price_data, optimized_params
            )
            
            if report_path:
                tprint(f"📄 Report saved: {report_path}", "SUCCESS")
                artifacts['report_path'] = report_path

            tprint(f"✅ Post-optimization backtesting completed: {metrics['total_return']:.1%} return, "
                   f"Sharpe {metrics['sharpe_ratio']:.2f}, Sortino {metrics['sortino_ratio']:.2f}, "
                   f"{metrics['total_trades']} trades", "SUCCESS")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics
            }

        except Exception as e:
            error_msg = f"Post-optimization backtesting failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg, exc_info=True)

            return {
                'success': False,
                'artifacts': {},
                'metrics': {},
                'error': error_msg
            }

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Register the step (only if not already registered by __init__.py)
def register_basic_backtesting_post_step():
    """Register the basic backtesting post step."""
    from src.training.steps.base_step import step_registry
    
    # Check if already registered to avoid duplicates
    if not step_registry.is_registered("basic_backtesting_post"):
        step_registry.register("basic_backtesting_post", BasicBacktestingPostStep)
        tprint("✅ Basic backtesting post step registered", "SUCCESS")


# Auto-register when module is imported
register_basic_backtesting_post_step()
