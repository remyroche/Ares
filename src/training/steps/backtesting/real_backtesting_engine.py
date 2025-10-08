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
from copy import deepcopy

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

# Import unified configuration
from .unified_config import UnifiedBacktestingConfig, ExecutionMode

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
    
    def __init__(self, config: UnifiedBacktestingConfig):
        """Initialize the real backtesting engine."""
        self.config = config
        self.logger = logger.getChild('RealBacktestingEngine')
        
        # Initialize data manager
        self.klines_manager = get_klines_manager(data_dir=config.data.data_dir)
        
        # Initialize hardware optimizers
        self.gpu_manager = get_m1_gpu_manager() if config.hardware.enable_gpu_acceleration else None
        self.memory_optimizer = get_m1_memory_optimizer() if config.hardware.enable_memory_optimization else None
        self.cpu_optimizer = get_m1_cpu_optimizer() if config.hardware.enable_parallel_processing else None
        
        # Initialize matrix operations
        self.matrix_ops = get_unified_matrix_operations()
        
        # Initialize ML utilities
        self.cv_validator = CVLSAValidator() if config.validation.enable_cv_validation else None
        self.hpo_optimizer = HyperparameterOptimizer() if config.validation.enable_hpo else None
        
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
        self.logger.info(f"📊 Loading market data for {self.config.data.symbol} on {self.config.data.exchange}")
        
        try:
            # Parse date range
            start_date = None
            end_date = None
            if self.config.data.start_date:
                start_date = datetime.strptime(self.config.data.start_date, '%Y-%m-%d')
            if self.config.data.end_date:
                end_date = datetime.strptime(self.config.data.end_date, '%Y-%m-%d')
            
            # Load data with memory optimization
            if self.memory_optimizer:
                with self.memory_optimizer.optimize_for_workload("data_loading"):
                    data = self.klines_manager.read_data(
                        symbol=self.config.data.symbol,
                        interval=self.config.data.timeframe,
                        data_type=self.config.data.data_type,
                        start_date=start_date,
                        end_date=end_date
                    )
            else:
                data = self.klines_manager.read_data(
                    symbol=self.config.data.symbol,
                    interval=self.config.data.timeframe,
                    data_type=self.config.data.data_type,
                    start_date=start_date,
                    end_date=end_date
                )
            
            if data is None or data.empty:
                raise ValueError(f"No data found for {self.config.data.symbol} on {self.config.data.exchange}")
            
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

            # Volatility features for trailing TP simulations
            returns = data['close'].pct_change().fillna(0)
            data['realized_volatility'] = returns.rolling(window=20).std().fillna(0) * np.sqrt(252)
            long_term_vol = data['realized_volatility'].rolling(window=100).mean()
            vol_std = data['realized_volatility'].rolling(window=100).std()
            data['volatility_zscore'] = ((data['realized_volatility'] - long_term_vol) / vol_std)
            data['volatility_zscore'] = data['volatility_zscore'].replace([np.inf, -np.inf], 0).fillna(0)
            data['volatility_bucket'] = pd.cut(
                data['realized_volatility'],
                bins=[-np.inf, 0.01, 0.03, np.inf],
                labels=['low', 'normal', 'high']
            )
            data['volatility_bucket'] = data['volatility_bucket'].astype(str).replace('nan', 'normal')

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
                base_size = self.config.backtesting.min_position_size + (confidence - 0.5) * (self.config.backtesting.max_position_size - self.config.backtesting.min_position_size)
                
                # Risk adjustment based on volatility
                if 'atr' in data.columns and i > 0:
                    volatility = data['atr'].iloc[i] / data['close'].iloc[i]
                    risk_adjusted_size = base_size * (1 - volatility)  # Reduce size in high volatility
                    positions.iloc[i] = np.clip(risk_adjusted_size, self.config.backtesting.min_position_size, self.config.backtesting.max_position_size)
                else:
                    positions.iloc[i] = base_size
        
        return positions
    

    async def execute_backtest(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        """Execute the actual backtest with trailing TP simulations."""
        self.logger.info("🚀 Executing backtest")

        try:
            scenario_configs = self._get_volatility_scenarios()
            base_scenario_name = 'normal' if 'normal' in scenario_configs else next(iter(scenario_configs.keys()), 'base')
            base_config = scenario_configs.get(base_scenario_name, {})

            base_result = self._run_backtest_simulation(data, signals, base_scenario_name, base_config)
            base_metrics = base_result['metrics']
            noise_sensitivity = self._estimate_noise_sensitivity(
                data,
                signals,
                base_scenario_name,
                base_config,
                base_metrics.get('total_return', 0.0)
            )
            base_metrics['noise_sensitivity'] = noise_sensitivity

            regime_performance = self._calculate_regime_performance(base_result['trade_log'])
            base_metrics['regime_performance'] = regime_performance

            self.equity_curve = base_result['equity_curve']
            self.trade_log = base_result['trade_log']
            self.performance_metrics = base_metrics

            persisted_path = self._persist_regime_performance(regime_performance)
            if persisted_path:
                self.logger.info(f"💾 Saved per-regime performance metrics to {persisted_path}")

            remaining_scenarios = {k: v for k, v in scenario_configs.items() if k != base_scenario_name}
            trial_results = self.simulate_trailing_tp_trials(data, signals, remaining_scenarios)

            if trial_results:
                self.performance_metrics['trailing_tp_trials'] = {
                    trial['scenario']: trial['metrics'] for trial in trial_results
                }

            self.logger.info(
                "✅ Backtest completed: %d exit trades, %.2f%% return (scenario=%s)",
                base_metrics.get('total_trades', 0),
                base_metrics.get('total_return', 0.0) * 100,
                base_scenario_name,
            )

            return {
                'performance_metrics': self.performance_metrics,
                'trade_log': self.trade_log,
                'equity_curve': self.equity_curve,
                'regime_performance': regime_performance,
                'trailing_tp_trials': trial_results,
                'final_portfolio_value': self.equity_curve[-1] if self.equity_curve else 0.0,
                'total_trades': self.performance_metrics.get('total_trades', 0)
            }

        except Exception as e:
            self.logger.error(f"❌ Backtest execution failed: {e}")
            raise

    def _get_trailing_tp_settings(self) -> Dict[str, Any]:
        """Get trailing take-profit configuration."""
        settings = {
            'activation_rr': getattr(self.config.trailing_tp, 'activation_rr', 1.2),
            'trail_distance_pct': getattr(self.config.trailing_tp, 'trail_distance_pct', 0.01),
            'volatility_sensitivity': getattr(self.config.trailing_tp, 'volatility_sensitivity', 1.0),
            'max_latency_seconds': getattr(self.config.trailing_tp, 'max_latency_seconds', 120),
            'noise_levels': list(getattr(self.config.trailing_tp, 'noise_levels', [0.0005, 0.001])),
        }

        custom_settings = self.config.custom_params.get('trailing_tp', {})
        if isinstance(custom_settings, dict):
            settings.update(custom_settings)

        return settings

    def _get_volatility_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Combine default and custom volatility scenario configurations."""
        scenarios: Dict[str, Dict[str, Any]] = {}
        default_scenarios = getattr(self.config.scenario_sweep, 'scenarios', {})
        custom_scenarios = self.config.custom_params.get('volatility_scenarios', {})

        for name, base_cfg in default_scenarios.items():
            base_copy = deepcopy(base_cfg)
            if isinstance(custom_scenarios, dict):
                base_copy.update(custom_scenarios.get(name, {}))
            scenarios[name] = base_copy

        if isinstance(custom_scenarios, dict):
            for name, cfg in custom_scenarios.items():
                if name not in scenarios:
                    scenarios[name] = deepcopy(cfg)

        return scenarios

    def _detect_regime_column(self, data: pd.DataFrame) -> Optional[str]:
        """Detect a regime column in the dataset if present."""
        candidate_columns = ['regime', 'regime_id', 'primary_regime', 'hmm_regime', 'composite_cluster_id']
        for column in candidate_columns:
            if column in data.columns:
                return column
        return None

    def _run_backtest_simulation(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        scenario_name: str,
        scenario_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a single backtest simulation for the provided scenario."""
        trailing_settings = self._get_trailing_tp_settings()
        portfolio_value = self.config.backtesting.initial_capital
        cash = self.config.backtesting.initial_capital
        position_shares = 0.0
        position_cost_basis = 0.0
        equity_curve = [portfolio_value]
        trade_log: List[Dict[str, Any]] = []
        risk_reward_values: List[float] = []
        realized_profits: List[float] = []
        trade_durations: List[float] = []
        activation_latencies: List[float] = []

        regime_column = self._detect_regime_column(data)
        base_stop_loss_pct = self.config.backtesting.stop_loss
        take_profit_pct = self.config.backtesting.take_profit * scenario_config.get('tp_multiplier', 1.0)

        trailing_state: Dict[str, Any] = {
            'active': False,
            'highest_price': None,
            'activation_price': None,
            'activation_time': None,
            'entry_time': None,
            'entry_index': None,
            'entry_price': None,
            'take_profit_price': None,
        }

        for i in range(1, len(data)):
            current_price = float(data['close'].iloc[i])
            signal = signals['signal'].iloc[i]
            position_size = float(signals['position'].iloc[i])
            timestamp = data.index[i]
            regime_value = str(data[regime_column].iloc[i]) if regime_column else 'global'

            volatility = float(data['realized_volatility'].iloc[i]) if 'realized_volatility' in data.columns else 0.0
            volatility_adjustment = 1.0 + volatility * trailing_settings['volatility_sensitivity']
            dynamic_trail_pct = trailing_settings['trail_distance_pct'] * scenario_config.get('trail_multiplier', 1.0) * volatility_adjustment
            dynamic_trail_pct = float(np.clip(dynamic_trail_pct, 0.0005, 0.2))

            if signal != 0 and position_size > 0:
                trade_value = portfolio_value * position_size
                shares = trade_value / current_price
                commission = trade_value * self.config.backtesting.commission_rate
                slippage = trade_value * self.config.backtesting.slippage_rate
                total_cost = trade_value + commission + slippage

                if signal == 1 and cash >= total_cost:
                    shares_to_buy = shares
                    cost = shares_to_buy * current_price + commission + slippage

                    if cost <= cash:
                        cash -= cost
                        position_shares += shares_to_buy
                        position_cost_basis += cost

                        activation_multiplier = scenario_config.get('activation_multiplier', 1.0)
                        activation_rr = trailing_settings['activation_rr'] * activation_multiplier
                        activation_move = base_stop_loss_pct * activation_rr

                        trailing_state['highest_price'] = current_price
                        trailing_state['activation_price'] = current_price * (1 + activation_move / max(volatility_adjustment, 1e-6))
                        trailing_state['activation_time'] = None
                        trailing_state['entry_time'] = timestamp
                        trailing_state['entry_index'] = i
                        trailing_state['entry_price'] = current_price
                        trailing_state['take_profit_price'] = current_price * (1 + take_profit_pct)
                        trailing_state['active'] = False

                        trade_log.append({
                            'timestamp': timestamp,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': cost,
                            'scenario': scenario_name,
                            'regime': regime_value,
                        })

            exit_reason = None
            if position_shares > 0:
                trailing_state['highest_price'] = (
                    current_price if trailing_state['highest_price'] is None
                    else max(trailing_state['highest_price'], current_price)
                )

                if not trailing_state['active'] and trailing_state['activation_price'] is not None:
                    if current_price >= trailing_state['activation_price']:
                        trailing_state['active'] = True
                        trailing_state['activation_time'] = timestamp

                if trailing_state['active'] and trailing_state['highest_price'] is not None:
                    trail_stop_price = trailing_state['highest_price'] * (1 - dynamic_trail_pct)
                    if current_price <= trail_stop_price:
                        exit_reason = 'TP_TRAIL'

                if exit_reason is None and trailing_state['take_profit_price'] is not None:
                    if current_price >= trailing_state['take_profit_price']:
                        exit_reason = 'TAKE_PROFIT'

                if exit_reason is None and base_stop_loss_pct > 0 and trailing_state['entry_price'] is not None:
                    stop_price = trailing_state['entry_price'] * (1 - base_stop_loss_pct)
                    if current_price <= stop_price:
                        exit_reason = 'STOP_LOSS'

                if exit_reason is None and signal == -1:
                    exit_reason = 'SIGNAL_EXIT'

                if exit_reason:
                    shares_to_sell = position_shares
                    gross_proceeds = shares_to_sell * current_price
                    commission = gross_proceeds * self.config.backtesting.commission_rate
                    slippage = gross_proceeds * self.config.backtesting.slippage_rate
                    cash += gross_proceeds - commission - slippage

                    profit = gross_proceeds - commission - slippage - position_cost_basis
                    realized_profits.append(float(profit))
                    position_shares = 0.0
                    position_cost_basis = 0.0

                    risk_reward = 0.0
                    if trailing_state['entry_price'] and base_stop_loss_pct > 0:
                        risk = trailing_state['entry_price'] * base_stop_loss_pct
                        reward = current_price - trailing_state['entry_price']
                        risk_reward = reward / risk if risk > 0 else 0.0
                        risk_reward_values.append(float(risk_reward))

                    if isinstance(timestamp, pd.Timestamp) and trailing_state['entry_time'] is not None:
                        duration_seconds = float((timestamp - trailing_state['entry_time']).total_seconds())
                    else:
                        duration_seconds = float(i - (trailing_state['entry_index'] or i))
                    trade_durations.append(duration_seconds)

                    activation_latency = 0.0
                    if exit_reason == 'TP_TRAIL' and trailing_state['activation_time'] is not None:
                        if isinstance(timestamp, pd.Timestamp):
                            activation_latency = float((timestamp - trailing_state['activation_time']).total_seconds())
                        else:
                            activation_latency = float(i - (trailing_state['entry_index'] or i))
                        activation_latencies.append(activation_latency)

                    trade_log.append({
                        'timestamp': timestamp,
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': current_price,
                        'profit': float(profit),
                        'exit_reason': exit_reason,
                        'scenario': scenario_name,
                        'regime': regime_value,
                        'risk_reward_ratio': float(risk_reward),
                        'duration_seconds': duration_seconds,
                        'activation_latency_seconds': activation_latency,
                    })

                    trailing_state = {
                        'active': False,
                        'highest_price': None,
                        'activation_price': None,
                        'activation_time': None,
                        'entry_time': None,
                        'entry_index': None,
                        'entry_price': None,
                        'take_profit_price': None,
                    }

            portfolio_value = cash + (position_shares * current_price)
            equity_curve.append(portfolio_value)

        latency_metrics = {
            'average_trade_duration_seconds': float(np.mean(trade_durations)) if trade_durations else 0.0,
            'average_trailing_latency_seconds': float(np.mean(activation_latencies)) if activation_latencies else 0.0,
            'latency_buffer_seconds': float(scenario_config.get('latency_buffer_seconds', 0.0)),
        }
        latency_metrics['latency_seconds'] = (
            latency_metrics['average_trailing_latency_seconds'] + latency_metrics['latency_buffer_seconds']
        )

        performance_metrics = self._calculate_performance_metrics(
            equity_curve,
            trade_log,
            risk_reward_values=risk_reward_values,
            trade_profits=realized_profits,
            latency_metrics=latency_metrics,
        )
        performance_metrics['scenario'] = scenario_name
        performance_metrics['latency_seconds'] = latency_metrics['latency_seconds']
        performance_metrics['latency_metrics'] = latency_metrics

        return {
            'scenario': scenario_name,
            'metrics': performance_metrics,
            'trade_log': trade_log,
            'equity_curve': equity_curve,
        }

    def _estimate_noise_sensitivity(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        scenario_name: str,
        scenario_config: Dict[str, Any],
        base_total_return: float,
    ) -> float:
        """Estimate noise sensitivity by perturbing price series."""
        noise_levels = scenario_config.get('noise_levels', [])
        if not noise_levels:
            return 0.0

        sensitivities: List[float] = []
        for noise in noise_levels:
            noisy_data = data.copy()
            noise_series = np.random.normal(0, noise, len(noisy_data))
            for column in ['open', 'high', 'low', 'close']:
                if column in noisy_data.columns:
                    noisy_data[column] = noisy_data[column] * (1 + noise_series)

            scenario_result = self._run_backtest_simulation(noisy_data, signals, scenario_name, scenario_config)
            trial_return = scenario_result['metrics'].get('total_return', 0.0)
            sensitivities.append(abs(base_total_return - trial_return))

        return float(np.mean(sensitivities)) if sensitivities else 0.0

    def simulate_trailing_tp_trials(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        scenario_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Simulate additional trailing TP scenarios."""
        scenario_configs = scenario_configs or self._get_volatility_scenarios()
        trial_results: List[Dict[str, Any]] = []

        for scenario_name, scenario_config in scenario_configs.items():
            scenario_result = self._run_backtest_simulation(data, signals, scenario_name, scenario_config)
            metrics = scenario_result['metrics']
            noise_sensitivity = self._estimate_noise_sensitivity(
                data,
                signals,
                scenario_name,
                scenario_config,
                metrics.get('total_return', 0.0)
            )
            metrics['noise_sensitivity'] = noise_sensitivity

            trial_results.append({
                'scenario': scenario_name,
                'metrics': {
                    'risk_reward_ratio': metrics.get('risk_reward_ratio', 0.0),
                    'profit_factor': metrics.get('profit_factor', 0.0),
                    'latency_seconds': metrics.get('latency_seconds', 0.0),
                    'noise_sensitivity': metrics.get('noise_sensitivity', 0.0),
                    'total_return': metrics.get('total_return', 0.0),
                    'win_rate': metrics.get('win_rate', 0.0),
                },
                'latency_metrics': metrics.get('latency_metrics', {}),
                'trade_count': metrics.get('total_trades', 0),
                'equity_curve': scenario_result['equity_curve'],
                'trade_log': scenario_result['trade_log'],
            })

        return trial_results

    def _calculate_regime_performance(self, trade_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate performance metrics per regime."""
        regime_stats: Dict[str, Dict[str, Any]] = {}
        exit_trades = [trade for trade in trade_log if trade.get('action') == 'SELL']

        for trade in exit_trades:
            regime = str(trade.get('regime', 'global'))
            stats = regime_stats.setdefault(
                regime,
                {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit': 0.0,
                    'average_rr_values': [],
                    'profit_components': [],
                },
            )

            profit = float(trade.get('profit', 0.0))
            stats['total_trades'] += 1
            stats['total_profit'] += profit
            stats['profit_components'].append(profit)
            stats['average_rr_values'].append(float(trade.get('risk_reward_ratio', 0.0)))

            if profit > 0:
                stats['winning_trades'] += 1
            elif profit < 0:
                stats['losing_trades'] += 1

        finalized_stats: Dict[str, Dict[str, Any]] = {}
        for regime, stats in regime_stats.items():
            trades = stats['total_trades']
            win_rate = stats['winning_trades'] / trades if trades else 0.0
            avg_rr = float(np.mean(stats['average_rr_values'])) if stats['average_rr_values'] else 0.0
            gross_profit = sum(p for p in stats['profit_components'] if p > 0)
            gross_loss = abs(sum(p for p in stats['profit_components'] if p < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

            finalized_stats[regime] = {
                'total_trades': trades,
                'winning_trades': stats['winning_trades'],
                'losing_trades': stats['losing_trades'],
                'win_rate': win_rate,
                'total_profit': stats['total_profit'],
                'average_rr': avg_rr,
                'profit_factor': profit_factor,
            }

        return finalized_stats

    def _persist_regime_performance(self, regime_stats: Dict[str, Any]) -> Optional[Path]:
        """Persist per-regime performance metrics for downstream optimizers."""
        if not regime_stats:
            return None

        try:
            output_dir = Path(self.config.reporting.output_dir or 'reports') / 'backtesting'
            ensure_directory(str(output_dir))
            file_path = output_dir / 'per_regime_performance.json'

            serializable = {
                regime: {
                    key: float(value) if isinstance(value, (np.floating, float)) else value
                    for key, value in metrics.items()
                }
                for regime, metrics in regime_stats.items()
            }

            safe_json_dump(serializable, file_path)
            self.config.custom_params['regime_performance_path'] = str(file_path)
            return file_path
        except Exception as e:
            self.logger.error(f"❌ Failed to persist per-regime performance metrics: {e}")
            return None

    def _calculate_performance_metrics(
        self,
        equity_curve: List[float],
        trade_log: List[Dict],
        risk_reward_values: Optional[List[float]] = None,
        trade_profits: Optional[List[float]] = None,
        latency_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        try:
            if len(equity_curve) < 2:
                return {}

            equity_series = pd.Series(equity_curve)
            returns = equity_series.pct_change().dropna()

            total_return_raw = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
            volatility = returns.std() * np.sqrt(252)
            turnover_metrics = self._calculate_turnover_metrics(trade_log, equity_curve)

            total_return = total_return_raw - turnover_metrics['market_impact_cost']
            annualized_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_drawdown = drawdown.min()

            realized_profits = trade_profits
            if realized_profits is None:
                realized_profits = [float(t.get('profit', 0.0)) for t in trade_log if t.get('action') == 'SELL']

            winning_trades = [p for p in realized_profits if p > 0]
            losing_trades = [p for p in realized_profits if p < 0]
            total_trades = len(realized_profits)
            win_rate = len(winning_trades) / total_trades if total_trades else 0.0

            gross_profit = sum(winning_trades)
            gross_loss = abs(sum(losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

            avg_win = np.mean(winning_trades) if winning_trades else 0.0
            avg_loss = np.mean(losing_trades) if losing_trades else 0.0

            risk_reward_ratio = float(np.mean(risk_reward_values)) if risk_reward_values else 0.0

            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'risk_reward_ratio': risk_reward_ratio,
                'turnover': turnover_metrics['turnover'],
                'average_holding_period_days': turnover_metrics['average_holding_period_days'],
                'capacity_utilization': turnover_metrics['capacity_utilization'],
                'capacity_limit': turnover_metrics['capacity_limit'],
                'market_impact_cost': turnover_metrics['market_impact_cost'],
                'raw_total_return': total_return_raw,
            }

            if latency_metrics:
                metrics['latency_metrics'] = {
                    key: float(value) for key, value in latency_metrics.items()
                }

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Failed to calculate performance metrics: {e}")
            return {}

    def _calculate_turnover_metrics(
        self,
        trade_log: List[Dict[str, Any]],
        equity_curve: List[float]
    ) -> Dict[str, float]:
        """Calculate turnover, holding period, and capacity diagnostics."""
        capacity_limit = getattr(self.config.backtesting, 'capacity_limit', 1.0)
        impact_coefficient = getattr(self.config.backtesting, 'market_impact_coefficient', 0.0005)
        warning_threshold = getattr(self.config.backtesting, 'turnover_warning_threshold', 0.8)

        if not trade_log:
            return {
                'turnover': 0.0,
                'average_holding_period_days': 0.0,
                'capacity_utilization': 0.0,
                'capacity_limit': capacity_limit,
                'market_impact_cost': 0.0
            }

        total_notional = 0.0
        holding_periods: List[float] = []
        open_positions: List[Dict[str, Any]] = []

        sorted_trades = sorted(trade_log, key=lambda t: t.get('timestamp'))

        for trade in sorted_trades:
            price = float(trade.get('price', 0.0))
            shares = float(trade.get('shares', 0.0))
            total_notional += abs(price * shares)

            action = str(trade.get('action', '')).lower()
            if action == 'buy':
                open_positions.append(trade)
            elif action == 'sell' and open_positions:
                entry_trade = open_positions.pop(0)
                entry_time = entry_trade.get('timestamp')
                exit_time = trade.get('timestamp')

                if isinstance(entry_time, pd.Timestamp) and isinstance(exit_time, pd.Timestamp):
                    holding_period = max((exit_time - entry_time).total_seconds() / 86400, 0.0)
                    holding_periods.append(holding_period)

        initial_equity = float(equity_curve[0]) if equity_curve else getattr(self.config.backtesting, 'initial_capital', 1.0)
        final_equity = float(equity_curve[-1]) if equity_curve else initial_equity
        average_equity = (initial_equity + final_equity) / 2 if final_equity > 0 else initial_equity
        turnover = total_notional / average_equity if average_equity > 0 else 0.0

        capacity_utilization = turnover / capacity_limit if capacity_limit else turnover
        market_impact_cost = turnover * impact_coefficient

        if capacity_limit:
            if capacity_utilization > 1.0:
                self.logger.warning(
                    "⚠️ Backtest capacity limit exceeded: %.2f%% utilization",
                    capacity_utilization * 100
                )
            elif capacity_utilization > warning_threshold:
                self.logger.warning(
                    "⚠️ Backtest capacity utilization high: %.2f%% of limit",
                    capacity_utilization * 100
                )

        average_holding_period = float(np.mean(holding_periods)) if holding_periods else 0.0

        return {
            'turnover': turnover,
            'average_holding_period_days': average_holding_period,
            'capacity_utilization': capacity_utilization,
            'capacity_limit': capacity_limit,
            'market_impact_cost': market_impact_cost
        }

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
    from .unified_config import create_config
    
    config = (create_config()
              .set_symbol(symbol)
              .set_exchange(exchange)
              .set_timeframe(timeframe)
              .set_data_dir(data_dir)
              .set_date_range(start_date or "2024-01-01", end_date or "2024-01-31")
              .set_custom_params(**kwargs)
              .build())
    
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