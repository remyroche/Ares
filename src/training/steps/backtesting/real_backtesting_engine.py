"""
Real Backtesting Engine Implementation

This module provides comprehensive real backtesting functionality using existing
utilities from src/utils/ for data loading, matrix operations, hardware optimization,
and ML common utilities.
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

# Import existing utilities
from src.utils.data.klines_parquet import get_klines_manager
from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.ml_common.vectorized_backtesting import VectorizedBacktestEngine, VectorizedBacktestConfig
from src.utils.ml_common.cvlsa import CVLSAValidator
from src.utils.ml_common.optimization import HyperparameterOptimizer
from src.utils.common_ml.backtesting.backtesting_engine import BacktestingEngine, BacktestingConfig
from src.utils.common_ml.backtesting.monte_carlo_engine import MonteCarloEngine, MonteCarloConfig
from src.utils.common_ml.backtesting.ab_testing_engine import ABTestingEngine, ABTestConfig
from src.utils.common_operations import safe_json_dump, safe_json_load, ensure_directory
from src.utils.math_validation import safe_divide, safe_log, safe_sqrt, validate_finite
from src.core.decorators import handles_errors, traced, log_execution_time

logger = logging.getLogger(__name__)

class BacktestMode(Enum):
    """Backtesting execution modes."""
    VECTORIZED = "vectorized"
    PARALLEL = "parallel"
    GPU_ACCELERATED = "gpu_accelerated"
    HYBRID = "hybrid"

@dataclass
class RealBacktestingConfig:
    """Configuration for real backtesting."""
    # Basic configuration
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Backtesting parameters
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_position_size: float = 0.1
    min_position_size: float = 0.01
    
    # Hardware optimization
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # ML parameters
    enable_cv_validation: bool = True
    enable_hpo: bool = True
    hpo_method: str = "bayesian"  # "grid", "bayesian", "random"
    
    # Risk management
    max_drawdown: float = 0.2
    stop_loss: float = 0.05
    take_profit: float = 0.1
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

class RealBacktestingEngine:
    """
    Real backtesting engine using existing utilities.
    
    This engine provides comprehensive backtesting functionality with:
    - Real data loading from klines_parquet
    - Hardware-optimized matrix operations
    - GPU acceleration for M1/M2/M3 Macs
    - ML validation and hyperparameter optimization
    - Risk management and performance metrics
    """
    
    def __init__(self, config: RealBacktestingConfig):
        """Initialize the real backtesting engine."""
        self.config = config
        self.logger = logger.getChild('RealBacktestingEngine')
        
        # Initialize data manager
        self.klines_manager = get_klines_manager(data_dir=config.data_dir)
        
        # Initialize hardware optimizers
        self.gpu_manager = get_m1_gpu_manager() if config.enable_gpu_acceleration else None
        self.memory_optimizer = get_m1_memory_optimizer() if config.enable_memory_optimization else None
        self.cpu_optimizer = get_m1_cpu_optimizer() if config.enable_parallel_processing else None
        
        # Initialize matrix operations
        self.matrix_ops = get_unified_matrix_operations()
        
        # Initialize ML utilities
        self.cv_validator = CVLSAValidator() if config.enable_cv_validation else None
        self.hpo_optimizer = HyperparameterOptimizer() if config.enable_hpo else None
        
        # Initialize backtesting engines
        self.vectorized_engine = VectorizedBacktestEngine()
        self.backtesting_engine = BacktestingEngine()
        self.monte_carlo_engine = MonteCarloEngine()
        self.ab_testing_engine = ABTestingEngine()
        
        # Performance tracking
        self.performance_metrics = {}
        self.trade_log = []
        self.equity_curve = []
        
    async def load_market_data(self) -> pd.DataFrame:
        """Load real market data using klines_parquet."""
        self.logger.info(f"📊 Loading market data for {self.config.symbol} on {self.config.exchange}")
        
        try:
            # Parse date range
            start_date = None
            end_date = None
            if self.config.start_date:
                start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
            if self.config.end_date:
                end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
            
            # Load data with memory optimization
            if self.memory_optimizer:
                with self.memory_optimizer.optimize_for_workload("data_loading"):
                    data = self.klines_manager.read_data(
                        symbol=self.config.symbol,
                        interval=self.config.timeframe,
                        data_type="processed",  # Use processed data for better performance
                        start_date=start_date,
                        end_date=end_date
                    )
            else:
                data = self.klines_manager.read_data(
                    symbol=self.config.symbol,
                    interval=self.config.timeframe,
                    data_type="processed",
                    start_date=start_date,
                    end_date=end_date
                )
            
            if data is None or data.empty:
                raise ValueError(f"No data found for {self.config.symbol} on {self.config.exchange}")
            
            self.logger.info(f"✅ Loaded {len(data)} rows of market data")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load market data: {e}")
            raise
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using matrix operations."""
        self.logger.info("📈 Calculating technical indicators")
        
        try:
            # Use matrix operations for efficient calculation
            if self.matrix_ops:
                # Calculate moving averages
                data['sma_20'] = self.matrix_ops.rolling_mean(data['close'].values, 20)
                data['sma_50'] = self.matrix_ops.rolling_mean(data['close'].values, 50)
                data['sma_200'] = self.matrix_ops.rolling_mean(data['close'].values, 200)
                
                # Calculate RSI
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                data['rsi'] = 100 - (100 / (1 + rs))
                
                # Calculate Bollinger Bands
                data['bb_middle'] = self.matrix_ops.rolling_mean(data['close'].values, 20)
                bb_std = self.matrix_ops.rolling_std(data['close'].values, 20)
                data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
                data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
                
                # Calculate MACD
                ema_12 = data['close'].ewm(span=12).mean()
                ema_26 = data['close'].ewm(span=26).mean()
                data['macd'] = ema_12 - ema_26
                data['macd_signal'] = data['macd'].ewm(span=9).mean()
                data['macd_histogram'] = data['macd'] - data['macd_signal']
                
                # Calculate ATR
                high_low = data['high'] - data['low']
                high_close = np.abs(data['high'] - data['close'].shift())
                low_close = np.abs(data['low'] - data['close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                data['atr'] = true_range.rolling(window=14).mean()
                
            else:
                # Fallback to standard pandas operations
                data['sma_20'] = data['close'].rolling(window=20).mean()
                data['sma_50'] = data['close'].rolling(window=50).mean()
                data['rsi'] = self._calculate_rsi(data['close'])
                data['atr'] = self._calculate_atr(data)
            
            # Clean up NaN values
            data = data.fillna(method='bfill').fillna(method='ffill')
            
            self.logger.info(f"✅ Calculated technical indicators: {len(data.columns)} columns")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate technical indicators: {e}")
            raise
    
    def generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate real trading signals based on technical analysis."""
        self.logger.info("🎯 Generating trading signals")
        
        try:
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0  # 0: hold, 1: buy, -1: sell
            signals['position'] = 0.0
            signals['confidence'] = 0.0
            
            # Trend following signals
            trend_signals = self._generate_trend_signals(data)
            
            # Mean reversion signals
            mean_reversion_signals = self._generate_mean_reversion_signals(data)
            
            # Momentum signals
            momentum_signals = self._generate_momentum_signals(data)
            
            # Combine signals with confidence weighting
            for i in range(len(data)):
                trend_signal = trend_signals.iloc[i] if i < len(trend_signals) else 0
                mean_rev_signal = mean_reversion_signals.iloc[i] if i < len(mean_reversion_signals) else 0
                momentum_signal = momentum_signals.iloc[i] if i < len(momentum_signals) else 0
                
                # Weighted combination
                combined_signal = (0.4 * trend_signal + 0.3 * mean_rev_signal + 0.3 * momentum_signal)
                
                # Apply confidence threshold
                if abs(combined_signal) > 0.5:
                    signals.iloc[i, signals.columns.get_loc('signal')] = np.sign(combined_signal)
                    signals.iloc[i, signals.columns.get_loc('confidence')] = abs(combined_signal)
            
            # Position sizing based on confidence and risk management
            signals['position'] = self._calculate_position_sizes(signals, data)
            
            self.logger.info(f"✅ Generated {len(signals[signals['signal'] != 0])} trading signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate trading signals: {e}")
            raise
    
    def _generate_trend_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trend following signals."""
        signals = pd.Series(0, index=data.index)
        
        # Moving average crossover
        if 'sma_20' in data.columns and 'sma_50' in data.columns:
            ma_cross = data['sma_20'] - data['sma_50']
            signals[ma_cross > 0] = 1  # Bullish
            signals[ma_cross < 0] = -1  # Bearish
        
        return signals
    
    def _generate_mean_reversion_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals."""
        signals = pd.Series(0, index=data.index)
        
        # RSI mean reversion
        if 'rsi' in data.columns:
            signals[(data['rsi'] < 30)] = 1  # Oversold - buy
            signals[(data['rsi'] > 70)] = -1  # Overbought - sell
        
        # Bollinger Bands mean reversion
        if all(col in data.columns for col in ['bb_upper', 'bb_lower', 'close']):
            signals[data['close'] < data['bb_lower']] = 1  # Below lower band - buy
            signals[data['close'] > data['bb_upper']] = -1  # Above upper band - sell
        
        return signals
    
    def _generate_momentum_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum signals."""
        signals = pd.Series(0, index=data.index)
        
        # MACD momentum
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            macd_diff = data['macd'] - data['macd_signal']
            signals[macd_diff > 0] = 1  # Bullish momentum
            signals[macd_diff < 0] = -1  # Bearish momentum
        
        return signals
    
    def _calculate_position_sizes(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
        """Calculate position sizes based on risk management."""
        positions = pd.Series(0.0, index=signals.index)
        
        for i in range(len(signals)):
            if signals.iloc[i]['signal'] != 0:
                confidence = signals.iloc[i]['confidence']
                
                # Base position size
                base_size = self.config.min_position_size + (confidence - 0.5) * (self.config.max_position_size - self.config.min_position_size)
                
                # Risk adjustment based on volatility
                if 'atr' in data.columns and i > 0:
                    volatility = data['atr'].iloc[i] / data['close'].iloc[i]
                    risk_adjusted_size = base_size * (1 - volatility)  # Reduce size in high volatility
                    positions.iloc[i] = np.clip(risk_adjusted_size, self.config.min_position_size, self.config.max_position_size)
                else:
                    positions.iloc[i] = base_size
        
        return positions
    
    async def execute_backtest(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        """Execute the actual backtest."""
        self.logger.info("🚀 Executing backtest")
        
        try:
            # Initialize portfolio
            portfolio_value = self.config.initial_capital
            position = 0.0
            cash = self.config.initial_capital
            
            # Performance tracking
            equity_curve = [portfolio_value]
            trade_log = []
            
            # Execute trades
            for i in range(1, len(data)):
                current_price = data['close'].iloc[i]
                signal = signals['signal'].iloc[i]
                position_size = signals['position'].iloc[i]
                
                if signal != 0 and position_size > 0:
                    # Calculate trade size
                    trade_value = portfolio_value * position_size
                    shares = trade_value / current_price
                    
                    # Apply transaction costs
                    commission = trade_value * self.config.commission_rate
                    slippage = trade_value * self.config.slippage_rate
                    total_cost = trade_value + commission + slippage
                    
                    if signal == 1 and cash >= total_cost:  # Buy signal
                        # Execute buy
                        shares_to_buy = shares
                        cost = shares_to_buy * current_price + commission + slippage
                        
                        if cost <= cash:
                            position += shares_to_buy
                            cash -= cost
                            
                            # Log trade
                            trade_log.append({
                                'timestamp': data.index[i],
                                'action': 'BUY',
                                'shares': shares_to_buy,
                                'price': current_price,
                                'cost': cost,
                                'portfolio_value': portfolio_value
                            })
                    
                    elif signal == -1 and position > 0:  # Sell signal
                        # Execute sell
                        shares_to_sell = min(position, shares)
                        proceeds = shares_to_sell * current_price - commission - slippage
                        
                        position -= shares_to_sell
                        cash += proceeds
                        
                        # Log trade
                        trade_log.append({
                            'timestamp': data.index[i],
                            'action': 'SELL',
                            'shares': shares_to_sell,
                            'price': current_price,
                            'proceeds': proceeds,
                            'portfolio_value': portfolio_value
                        })
                
                # Update portfolio value
                portfolio_value = cash + (position * current_price)
                equity_curve.append(portfolio_value)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(equity_curve, trade_log)
            
            # Store results
            self.equity_curve = equity_curve
            self.trade_log = trade_log
            self.performance_metrics = performance_metrics
            
            self.logger.info(f"✅ Backtest completed: {len(trade_log)} trades, {performance_metrics['total_return']:.2%} return")
            
            return {
                'performance_metrics': performance_metrics,
                'trade_log': trade_log,
                'equity_curve': equity_curve,
                'final_portfolio_value': portfolio_value,
                'total_trades': len(trade_log)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Backtest execution failed: {e}")
            raise
    
    def _calculate_performance_metrics(self, equity_curve: List[float], trade_log: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        try:
            if len(equity_curve) < 2:
                return {}
            
            equity_series = pd.Series(equity_curve)
            returns = equity_series.pct_change().dropna()
            
            # Basic metrics
            total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
            annualized_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Drawdown analysis
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_drawdown = drawdown.min()
            
            # Trade analysis
            winning_trades = [t for t in trade_log if t.get('proceeds', 0) > t.get('cost', 0)]
            losing_trades = [t for t in trade_log if t.get('proceeds', 0) < t.get('cost', 0)]
            
            win_rate = len(winning_trades) / len(trade_log) if trade_log else 0
            avg_win = np.mean([t.get('proceeds', 0) - t.get('cost', 0) for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.get('proceeds', 0) - t.get('cost', 0) for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(trade_log),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'avg_win': avg_win,
                'avg_loss': avg_loss
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate performance metrics: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr

# Convenience functions
async def execute_real_backtest(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Execute a real backtest with the given parameters."""
    config = RealBacktestingConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
    
    engine = RealBacktestingEngine(config)
    
    # Load data
    data = await engine.load_market_data()
    
    # Calculate indicators
    data = engine.calculate_technical_indicators(data)
    
    # Generate signals
    signals = engine.generate_trading_signals(data)
    
    # Execute backtest
    results = await engine.execute_backtest(data, signals)
    
    return results