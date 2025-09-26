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
from src.utils.nas_tas.backtesting_engine import BacktestingEngine, BacktestingConfig
from src.utils.nas_tas.monte_carlo_engine import UnifiedMonteCarloEngine as MonteCarloEngine, MonteCarloConfig
from src.utils.common_ml.backtesting.ab_testing_engine import ABTestingEngine, ABTestConfig
from src.utils.common_operations import safe_json_dump, safe_json_load, ensure_directory
from src.utils.math_validation import safe_divide, safe_log, safe_sqrt, validate_finite
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.tactician.position_sizer import PositionSizer
from src.tactician.leverage_sizer import LeverageSizer

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

        # Signal activation threshold (weights are now optimized upstream)
        self.signal_weight_threshold: float = float(
            self.config.custom_params.get('signal_activation_threshold', 0.5)
        )

        # Cached artifacts and tactician components
        self.artifact_manager = get_artifact_manager()
        self._artifact_signal_payload = self._load_latest_signal_weight_artifact()
        self.optimized_signal_weights: Optional[Dict[str, float]] = None
        if isinstance(self._artifact_signal_payload, dict):
            weights_payload = self._artifact_signal_payload.get('weights')
            if isinstance(weights_payload, dict):
                self.optimized_signal_weights = weights_payload
            elif all(isinstance(value, (int, float)) for value in self._artifact_signal_payload.values()):
                self.optimized_signal_weights = {
                    key: float(value)
                    for key, value in self._artifact_signal_payload.items()
                }

            threshold = self._artifact_signal_payload.get('threshold')
            if (
                threshold is not None
                and 'signal_activation_threshold' not in self.config.custom_params
            ):
                try:
                    self.signal_weight_threshold = float(threshold)
                except (TypeError, ValueError):
                    self.logger.debug(
                        "Ignoring non-numeric activation threshold from artifact: %s",
                        threshold,
                    )

        self.position_sizer: Optional[PositionSizer] = None
        self.leverage_sizer: Optional[LeverageSizer] = None
        self._tactician_initialized = False
        self._logged_signal_weight_source = False
        
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
            
            # Clean up NaN values
            data = data.fillna(method='bfill').fillna(method='ffill')
            
            self.logger.info(f"✅ Calculated technical indicators: {len(data.columns)} columns")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate technical indicators: {e}")
            raise
    
    async def generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate real trading signals using optimized weights and tactician sizing."""
        self.logger.info("🎯 Generating trading signals")

        try:
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0  # 0: hold, 1: buy, -1: sell
            signals['position'] = 0.0
            signals['confidence'] = 0.0

            trend_signals = self._generate_trend_signals(data)
            mean_reversion_signals = self._generate_mean_reversion_signals(data)
            momentum_signals = self._generate_momentum_signals(data)

            component_frame = pd.DataFrame({
                'trend': trend_signals.reindex(data.index, fill_value=0.0),
                'mean_reversion': mean_reversion_signals.reindex(data.index, fill_value=0.0),
                'momentum': momentum_signals.reindex(data.index, fill_value=0.0),
            })

            signal_weights = self._resolve_signal_weights()

            combined_signal_series = (
                signal_weights['trend'] * component_frame['trend']
                + signal_weights['mean_reversion'] * component_frame['mean_reversion']
                + signal_weights['momentum'] * component_frame['momentum']
            )

            activation_threshold = self.signal_weight_threshold
            active_mask = combined_signal_series.abs() > activation_threshold
            signals.loc[active_mask, 'signal'] = np.sign(combined_signal_series[active_mask])
            signals.loc[active_mask, 'confidence'] = combined_signal_series[active_mask].abs()

            positions, leverages = await self._calculate_position_and_leverage(
                signals,
                data,
                component_frame,
                combined_signal_series,
                signal_weights,
            )

            signals['position'] = positions
            signals['leverage'] = leverages

            self.logger.info(
                "✅ Generated %d trading signals with tactician sizing",
                int(active_mask.sum()),
            )
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

    def _normalize_signal_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize provided signal weights so they sum to 1."""

        trend = float(weights.get('trend', 0.0))
        mean_rev = float(weights.get('mean_reversion', 0.0))
        momentum = float(weights.get('momentum', 0.0))
        weight_array = np.array([trend, mean_rev, momentum])
        positive_mask = weight_array > 0

        if not positive_mask.any():
            return {'trend': 1 / 3, 'mean_reversion': 1 / 3, 'momentum': 1 / 3}

        normalized = weight_array / weight_array.sum()
        return {
            'trend': float(normalized[0]),
            'mean_reversion': float(normalized[1]),
            'momentum': float(normalized[2])
        }
    
    def _generate_momentum_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum signals."""
        signals = pd.Series(0, index=data.index)
        
        # MACD momentum
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            macd_diff = data['macd'] - data['macd_signal']
            signals[macd_diff > 0] = 1  # Bullish momentum
            signals[macd_diff < 0] = -1  # Bearish momentum
        
        return signals
    
    async def _calculate_position_and_leverage(
        self,
        signals: pd.DataFrame,
        data: pd.DataFrame,
        component_signals: pd.DataFrame,
        combined_signal: pd.Series,
        signal_weights: Dict[str, float],
    ) -> Tuple[pd.Series, pd.Series]:
        """Use Tactician sizing components to determine position size and leverage."""

        await self._ensure_tactician_components()

        default_positions = pd.Series(0.0, index=signals.index)
        default_leverage = pd.Series(1.0, index=signals.index)

        if not self._tactician_initialized or not self.position_sizer or not self.leverage_sizer:
            self.logger.debug("Tactician components unavailable, falling back to default sizing")
            return default_positions, default_leverage

        min_size = float(self.config.backtesting.min_position_size)
        max_size = float(self.config.backtesting.max_position_size)

        atr_ratio: Optional[pd.Series] = None
        if 'atr' in data.columns and 'close' in data.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                atr_ratio = (data['atr'] / data['close']).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                atr_ratio = atr_ratio.clip(lower=0.0)

        positions: List[float] = []
        leverages: List[float] = []

        for idx in range(len(signals)):
            if signals['signal'].iat[idx] == 0:
                positions.append(0.0)
                leverages.append(1.0)
                continue

            price = float(data['close'].iat[idx])
            combined_confidence = float(np.clip(abs(combined_signal.iat[idx]), 0.0, 1.0))
            intensity = combined_confidence
            risk_component = float(atr_ratio.iat[idx]) if atr_ratio is not None else 0.2
            reliability = float(np.clip(1.0 - risk_component, 0.0, 1.0))

            price_target_confidences = {
                'trend': float(np.clip(abs(component_signals['trend'].iat[idx]), 0.0, 1.0)),
                'mean_reversion': float(np.clip(abs(component_signals['mean_reversion'].iat[idx]), 0.0, 1.0)),
                'momentum': float(np.clip(abs(component_signals['momentum'].iat[idx]), 0.0, 1.0)),
            }
            adversarial_confidences = {
                key: float(np.clip(1.0 - value, 0.0, 1.0))
                for key, value in price_target_confidences.items()
            }
            directional_confidence = {
                'trend_direction': float(np.clip(component_signals['trend'].iat[idx], -1.0, 1.0)),
                'mean_reversion_direction': float(np.clip(component_signals['mean_reversion'].iat[idx], -1.0, 1.0)),
                'momentum_direction': float(np.clip(component_signals['momentum'].iat[idx], -1.0, 1.0)),
            }

            ml_predictions = {
                'combined_confidence': combined_confidence,
                'price_target_confidences': price_target_confidences,
                'adversarial_confidences': adversarial_confidences,
                'directional_confidence': directional_confidence,
                'intensity': intensity,
                'reliability': reliability,
                'risk_score': float(np.clip(risk_component, 0.0, 1.0)),
            }

            try:
                position_analysis = await self.position_sizer.calculate_position_size(
                    ml_predictions=ml_predictions,
                    current_price=price,
                    account_balance=self.config.backtesting.initial_capital,
                    analyst_confidence=combined_confidence,
                    tactician_confidence=combined_confidence,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.debug("Position sizing fallback due to error: %s", exc, exc_info=True)
                position_analysis = None

            position_size = None
            if position_analysis:
                position_size = position_analysis.get('final_position_size')

            if position_size is None:
                span = max_size - min_size
                position_size = min_size + span * combined_confidence

            position_size = float(np.clip(position_size, min_size, max_size))
            positions.append(position_size)

            try:
                leverage_analysis = await self.leverage_sizer.calculate_leverage(
                    ml_predictions=ml_predictions,
                    current_price=price,
                    account_balance=self.config.backtesting.initial_capital,
                    analyst_confidence=combined_confidence,
                    tactician_confidence=combined_confidence,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.debug("Leverage sizing fallback due to error: %s", exc, exc_info=True)
                leverage_analysis = None

            leverage_value = None
            if leverage_analysis:
                leverage_value = leverage_analysis.get('final_leverage')

            if leverage_value is None or not np.isfinite(leverage_value):
                leverage_bias = max(0.0, signal_weights['momentum'] - signal_weights['mean_reversion'])
                leverage_value = 1.0 + leverage_bias

            leverages.append(float(max(1.0, leverage_value)))

        return pd.Series(positions, index=signals.index), pd.Series(leverages, index=signals.index)

    def _resolve_signal_weights(self) -> Dict[str, float]:
        """Determine which signal weights to use for aggregation."""

        default_weights = {'trend': 1 / 3, 'mean_reversion': 1 / 3, 'momentum': 1 / 3}
        source = 'default'
        weights = default_weights

        optimized_weights = self.config.custom_params.get('optimized_signal_weights')
        custom_weights = self.config.custom_params.get('signal_weights')

        if optimized_weights:
            weights = self._normalize_signal_weights(optimized_weights)
            source = 'custom_params.optimized_signal_weights'
        elif custom_weights:
            weights = self._normalize_signal_weights(custom_weights)
            source = 'custom_params.signal_weights'
        elif self.optimized_signal_weights:
            weights = self._normalize_signal_weights(self.optimized_signal_weights)
            source = 'artifact.final_signal_weight_optimization'

        if not self._logged_signal_weight_source:
            self.logger.info("⚙️ Using %s for signal weights: %s", source, weights)
            self._logged_signal_weight_source = True

        return weights

    def _load_latest_signal_weight_artifact(self) -> Optional[Dict[str, Any]]:
        """Load cached optimized signal weights, if available."""

        try:
            data, _ = self.artifact_manager.load_most_recent_artifact(
                'final_signal_weight_optimization',
                directory='artifacts',
                extension='.json',
            )
            if isinstance(data, dict):
                return data
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.debug("No cached signal weight artifact available: %s", exc)

        return None

    async def _ensure_tactician_components(self) -> None:
        """Initialise tactician position and leverage sizers if needed."""

        if self._tactician_initialized:
            return

        try:
            tactician_config = self._build_tactician_config()
            self.position_sizer = PositionSizer(tactician_config)
            self.leverage_sizer = LeverageSizer(tactician_config)
            await asyncio.gather(
                self.position_sizer.initialize(),
                self.leverage_sizer.initialize(),
            )
            self._tactician_initialized = bool(
                self.position_sizer and self.position_sizer.is_initialized
                and self.leverage_sizer and self.leverage_sizer.is_initialized
            )
            if not self._tactician_initialized:
                self.logger.warning("Tactician sizing components failed to initialise; using fallback sizing")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to initialise tactician sizing components: %s", exc, exc_info=True)
            self.position_sizer = None
            self.leverage_sizer = None
            self._tactician_initialized = False

    def _build_tactician_config(self) -> Dict[str, Any]:
        """Merge configuration inputs for tactician sizing components."""

        custom_config = self.config.custom_params.get('tactician', {})
        custom_config = custom_config if isinstance(custom_config, dict) else {}

        base_config: Dict[str, Any] = {
            key: value
            for key, value in custom_config.items()
            if key not in {'position_sizing', 'leverage_sizing', 'step17_optimization'}
        }

        position_defaults = {
            'kelly_multiplier': 0.25,
            'max_position_size': float(self.config.backtesting.max_position_size),
            'min_position_size': float(self.config.backtesting.min_position_size),
            'confidence_threshold': 0.6,
            'positionsize_combined_threshold': 0.7,
            'ml_weight': 0.7,
        }
        leverage_defaults = {
            'min_leverage': 1.0,
            'max_leverage': 3.0,
            'leverage_combined_threshold': 0.75,
            'leverage_multiplier': 2.0,
        }

        custom_position = custom_config.get('position_sizing', {})
        custom_leverage = custom_config.get('leverage_sizing', {})
        custom_step17 = custom_config.get('step17_optimization', {})

        step17_config = self.config.custom_params.get('step17_optimization', {})
        final_parameters = self.config.custom_params.get('final_parameters', {})

        step17_position = step17_config.get('position_sizing', {})
        step17_leverage = step17_config.get('leverage', {})
        final_position = final_parameters.get('position_sizing', {})
        final_leverage = final_parameters.get('leverage', {})

        merged_position = {
            **position_defaults,
            **step17_position,
            **final_position,
            **custom_position,
        }
        merged_leverage = {
            **leverage_defaults,
            **step17_leverage,
            **final_leverage,
            **custom_leverage,
        }

        step17_merged = {
            'position_sizing': {**step17_position, **final_position, **custom_step17.get('position_sizing', {})},
            'leverage': {**step17_leverage, **final_leverage, **custom_step17.get('leverage', {})},
        }

        base_config['position_sizing'] = merged_position
        base_config['leverage_sizing'] = merged_leverage
        base_config['step17_optimization'] = step17_merged

        return base_config

    async def execute_backtest(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        """Execute the actual backtest."""
        self.logger.info("🚀 Executing backtest")

        try:
            initial_capital = float(self.config.backtesting.initial_capital)
            portfolio_value = initial_capital
            cash = initial_capital
            position = 0.0  # Shares currently held
            margin_loans = 0.0
            position_lots: List[Dict[str, float]] = []

            equity_curve = [portfolio_value]
            trade_log: List[Dict[str, Any]] = []

            min_size = float(self.config.backtesting.min_position_size)
            max_size = float(self.config.backtesting.max_position_size)

            commission_rate = float(self.config.backtesting.commission_rate)
            slippage_rate = float(self.config.backtesting.slippage_rate)

            for i in range(1, len(data)):
                current_price = float(data['close'].iloc[i])
                if current_price <= 0:
                    equity_curve.append(portfolio_value)
                    continue

                signal = int(signals['signal'].iloc[i])
                position_size = float(np.clip(signals['position'].iloc[i], min_size, max_size))
                leverage = float(signals['leverage'].iloc[i]) if 'leverage' in signals.columns else 1.0
                leverage = max(1.0, leverage)

                if signal == 1 and position_size > 0.0:
                    capital_required = portfolio_value * position_size
                    exposure = capital_required * leverage
                    commission = exposure * commission_rate
                    slippage = exposure * slippage_rate
                    total_outlay = capital_required + commission + slippage

                    if total_outlay <= cash and exposure > 0:
                        shares_to_buy = exposure / current_price
                        borrowed = max(0.0, exposure - capital_required)

                        position += shares_to_buy
                        cash -= total_outlay
                        margin_loans += borrowed
                        position_lots.append({
                            'shares': shares_to_buy,
                            'entry_price': current_price,
                            'capital_required': capital_required,
                            'borrowed': borrowed,
                            'leverage': leverage,
                        })

                        trade_log.append({
                            'timestamp': data.index[i],
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'capital_required': capital_required,
                            'borrowed': borrowed,
                            'exposure': exposure,
                            'commission': commission,
                            'slippage': slippage,
                            'portfolio_value': portfolio_value,
                            'leverage': leverage,
                        })

                elif signal == -1 and position > 0.0:
                    shares_to_sell = position
                    exposure = shares_to_sell * current_price
                    commission = exposure * commission_rate
                    slippage = exposure * slippage_rate

                    remaining_shares = shares_to_sell
                    capital_released = 0.0
                    borrowed_repaid = 0.0
                    realized_pnl = 0.0
                    leverage_accumulator = 0.0
                    leverage_shares = 0.0

                    while remaining_shares > 0 and position_lots:
                        lot = position_lots[0]
                        lot_shares = min(lot['shares'], remaining_shares)
                        if lot_shares <= 0:
                            position_lots.pop(0)
                            continue

                        proportion = lot_shares / lot['shares']
                        capital_released += lot['capital_required'] * proportion
                        borrowed_repaid += lot['borrowed'] * proportion
                        realized_pnl += (current_price - lot['entry_price']) * lot_shares
                        leverage_accumulator += lot['leverage'] * lot_shares
                        leverage_shares += lot_shares

                        lot['shares'] -= lot_shares
                        lot['capital_required'] -= lot['capital_required'] * proportion
                        lot['borrowed'] -= lot['borrowed'] * proportion

                        if lot['shares'] <= 1e-8:
                            position_lots.pop(0)

                        remaining_shares -= lot_shares

                    position = max(0.0, position - shares_to_sell)
                    margin_loans = max(0.0, margin_loans - borrowed_repaid)
                    cash += max(0.0, capital_released + realized_pnl - commission - slippage)

                    avg_leverage = leverage_accumulator / leverage_shares if leverage_shares > 0 else leverage

                    trade_log.append({
                        'timestamp': data.index[i],
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': current_price,
                        'capital_returned': capital_released,
                        'borrowed_repaid': borrowed_repaid,
                        'pnl': realized_pnl,
                        'commission': commission,
                        'slippage': slippage,
                        'portfolio_value': portfolio_value,
                        'leverage': avg_leverage,
                    })

                portfolio_value = cash + (position * current_price) - margin_loans
                equity_curve.append(portfolio_value)

            performance_metrics = self._calculate_performance_metrics(equity_curve, trade_log)

            self.equity_curve = equity_curve
            self.trade_log = trade_log
            self.performance_metrics = performance_metrics

            self.logger.info(
                "✅ Backtest completed: %d trades, %.2f%% return",
                len(trade_log),
                performance_metrics.get('total_return', 0.0) * 100,
            )

            return {
                'performance_metrics': performance_metrics,
                'trade_log': trade_log,
                'equity_curve': equity_curve,
                'final_portfolio_value': portfolio_value,
                'total_trades': len(trade_log),
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
            
            # Trade analysis based on realised (SELL) trades
            realised_trades = [t for t in trade_log if t.get('action') == 'SELL']
            winning_trades = [t for t in realised_trades if t.get('pnl', 0.0) > 0]
            losing_trades = [t for t in realised_trades if t.get('pnl', 0.0) < 0]

            win_rate = len(winning_trades) / len(realised_trades) if realised_trades else 0
            avg_win = np.mean([t.get('pnl', 0.0) for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.get('pnl', 0.0) for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(realised_trades),
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
    signals = await engine.generate_trading_signals(data)
    
    # Execute backtest
    results = await engine.execute_backtest(data, signals)
    
    return results