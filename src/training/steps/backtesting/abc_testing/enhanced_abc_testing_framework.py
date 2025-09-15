"""
Enhanced A/B/C Testing Framework with Multi-Model Support and Flexible TPSL

This module provides an enhanced A/B/C testing framework that supports:
- Unlimited number of models (not just 3)
- Flexible Take Profit/Stop Loss (TPSL) parameters
- Advanced TPSL strategies and configurations
- Dynamic TPSL adjustment based on market conditions
- TPSL parameter optimization and testing

Key Features:
- Support for 3+ models (A/B/C/D/E/F... testing)
- Multiple TPSL strategies (Fixed, ATR-based, Volatility-based, Dynamic)
- TPSL parameter grid search and optimization
- Real-time TPSL adjustment based on market conditions
- TPSL performance analysis and comparison
- Integration with existing framework components
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
from pathlib import Path
import copy
import uuid
from itertools import product
import json

# Import existing framework components
from .abc_testing_framework import ABCTestingFramework, ABCTestingConfig, ABCTestingResults
from .multi_model_orchestrator import MultiModelOrchestrator, ModelConfig, OrchestrationConfig
from .paper_trading_engine import PaperTradingEngine, PaperTradingConfig, MarketData, OrderSide, OrderType
from .risk_management import RiskManager, RiskLimits, PositionSizingConfig, PositionSizingMethod
from .statistical_analysis import StatisticalAnalyzer, StatisticalTestConfig, StatisticalResults
from .performance_monitoring import PerformanceMonitor, MonitoringConfig, AlertConfig
from .results_visualization import ResultsVisualizer, VisualizationConfig
from .configuration_management import ConfigurationManager, ConfigurationEntry, ConfigurationScope, ConfigurationFormat

logger = logging.getLogger(__name__)


class TPSLStrategy(Enum):
    """Take Profit/Stop Loss strategies."""
    FIXED = "fixed"                    # Fixed percentage TPSL
    ATR_BASED = "atr_based"           # Based on Average True Range
    VOLATILITY_BASED = "volatility_based"  # Based on asset volatility
    DYNAMIC = "dynamic"               # Dynamic adjustment based on market conditions
    TRAILING = "trailing"             # Trailing stop loss
    BREAKEVEN = "breakeven"           # Move to breakeven after profit target
    SCALING = "scaling"               # Scale out positions
    TIME_BASED = "time_based"         # Time-based TPSL
    MOMENTUM_BASED = "momentum_based" # Based on momentum indicators
    SUPPORT_RESISTANCE = "support_resistance"  # Based on support/resistance levels


class TPSLMode(Enum):
    """TPSL execution modes."""
    IMMEDIATE = "immediate"           # Execute immediately when triggered
    NEXT_BAR = "next_bar"            # Execute at next bar open
    END_OF_DAY = "end_of_day"        # Execute at end of day
    CONDITIONAL = "conditional"      # Execute based on additional conditions


@dataclass
class TPSLConfig:
    """Take Profit/Stop Loss configuration."""
    strategy: TPSLStrategy = TPSLStrategy.FIXED
    mode: TPSLMode = TPSLMode.IMMEDIATE
    
    # Fixed TPSL parameters
    take_profit_pct: float = 0.02     # 2% take profit
    stop_loss_pct: float = 0.01       # 1% stop loss
    
    # ATR-based parameters
    atr_multiplier_tp: float = 2.0    # 2x ATR for take profit
    atr_multiplier_sl: float = 1.0    # 1x ATR for stop loss
    atr_period: int = 14              # ATR calculation period
    
    # Volatility-based parameters
    volatility_multiplier_tp: float = 1.5  # 1.5x volatility for take profit
    volatility_multiplier_sl: float = 1.0  # 1x volatility for stop loss
    volatility_period: int = 20       # Volatility calculation period
    
    # Dynamic parameters
    dynamic_adjustment_factor: float = 0.5  # Adjustment sensitivity
    min_tp_pct: float = 0.005        # Minimum take profit (0.5%)
    max_tp_pct: float = 0.05         # Maximum take profit (5%)
    min_sl_pct: float = 0.005        # Minimum stop loss (0.5%)
    max_sl_pct: float = 0.03         # Maximum stop loss (3%)
    
    # Trailing parameters
    trailing_start_pct: float = 0.01  # Start trailing after 1% profit
    trailing_step_pct: float = 0.005  # Trailing step (0.5%)
    
    # Scaling parameters
    scale_out_levels: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])  # Scale out at 50%, 30%, 20%
    scale_out_sizes: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.5])  # Scale out sizes
    
    # Time-based parameters
    max_hold_time_hours: int = 24     # Maximum hold time in hours
    time_decay_factor: float = 0.1    # Time decay adjustment factor
    
    # Momentum parameters
    momentum_period: int = 10         # Momentum calculation period
    momentum_threshold: float = 0.5   # Momentum threshold for adjustment
    
    # Support/Resistance parameters
    sr_lookback: int = 20             # Support/resistance lookback period
    sr_buffer_pct: float = 0.002      # Buffer around S/R levels (0.2%)
    
    # Advanced features
    enable_breakeven: bool = True     # Enable breakeven functionality
    breakeven_trigger_pct: float = 0.01  # Trigger breakeven after 1% profit
    enable_partial_tp: bool = False   # Enable partial take profits
    enable_trailing_sl: bool = False  # Enable trailing stop loss
    enable_time_stop: bool = False    # Enable time-based stop
    
    # Risk management
    max_risk_per_trade: float = 0.02  # Maximum risk per trade (2%)
    min_risk_reward_ratio: float = 1.5  # Minimum risk-reward ratio
    
    # Market condition adjustments
    volatile_market_multiplier: float = 1.5  # Adjust for volatile markets
    trending_market_multiplier: float = 0.8  # Adjust for trending markets
    sideways_market_multiplier: float = 1.2  # Adjust for sideways markets


@dataclass
class TPSLResult:
    """TPSL execution result."""
    symbol: str
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    exit_reason: str  # "take_profit", "stop_loss", "time_stop", "manual", etc.
    tpsl_config: TPSLConfig
    profit_loss: float
    profit_loss_pct: float
    hold_time_hours: float
    max_profit_pct: float
    max_drawdown_pct: float
    risk_reward_ratio: float
    tpsl_performance_score: float


@dataclass
class TPSLOptimizationResult:
    """TPSL parameter optimization result."""
    best_config: TPSLConfig
    best_score: float
    optimization_results: List[Tuple[TPSLConfig, float]]
    parameter_importance: Dict[str, float]
    optimization_time: float
    total_tests: int


class TPSLManager:
    """Advanced Take Profit/Stop Loss management system."""
    
    def __init__(self, base_config: TPSLConfig):
        """Initialize TPSL manager."""
        self.base_config = base_config
        self.logger = logger.getChild('TPSLManager')
        
        # Active TPSL orders
        self.active_tpsl_orders: Dict[str, Dict[str, Any]] = {}
        
        # TPSL performance tracking
        self.tpsl_results: List[TPSLResult] = []
        self.tpsl_performance_metrics: Dict[str, float] = {}
        
        # Market condition tracking
        self.market_conditions: Dict[str, str] = {}  # "volatile", "trending", "sideways"
        
        self.logger.info("🚀 TPSLManager initialized")
        self.logger.info(f"📊 TPSL Strategy: {base_config.strategy.value}")
        self.logger.info(f"📊 Take Profit: {base_config.take_profit_pct:.1%}")
        self.logger.info(f"📊 Stop Loss: {base_config.stop_loss_pct:.1%}")
    
    def calculate_tpsl_levels(self, symbol: str, entry_price: float, 
                            market_data: MarketData, position_side: OrderSide) -> Tuple[float, float]:
        """Calculate take profit and stop loss levels."""
        try:
            config = self._get_dynamic_config(symbol, market_data)
            
            if config.strategy == TPSLStrategy.FIXED:
                return self._calculate_fixed_tpsl(entry_price, config, position_side)
            elif config.strategy == TPSLStrategy.ATR_BASED:
                return self._calculate_atr_tpsl(entry_price, market_data, config, position_side)
            elif config.strategy == TPSLStrategy.VOLATILITY_BASED:
                return self._calculate_volatility_tpsl(entry_price, market_data, config, position_side)
            elif config.strategy == TPSLStrategy.DYNAMIC:
                return self._calculate_dynamic_tpsl(entry_price, market_data, config, position_side)
            elif config.strategy == TPSLStrategy.TRAILING:
                return self._calculate_trailing_tpsl(entry_price, market_data, config, position_side)
            elif config.strategy == TPSLStrategy.SCALING:
                return self._calculate_scaling_tpsl(entry_price, market_data, config, position_side)
            elif config.strategy == TPSLStrategy.TIME_BASED:
                return self._calculate_time_tpsl(entry_price, market_data, config, position_side)
            elif config.strategy == TPSLStrategy.MOMENTUM_BASED:
                return self._calculate_momentum_tpsl(entry_price, market_data, config, position_side)
            elif config.strategy == TPSLStrategy.SUPPORT_RESISTANCE:
                return self._calculate_sr_tpsl(entry_price, market_data, config, position_side)
            else:
                return self._calculate_fixed_tpsl(entry_price, config, position_side)
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating TPSL levels: {e}")
            return self._calculate_fixed_tpsl(entry_price, self.base_config, position_side)
    
    def _get_dynamic_config(self, symbol: str, market_data: MarketData) -> TPSLConfig:
        """Get dynamic TPSL configuration based on market conditions."""
        config = copy.deepcopy(self.base_config)
        
        # Adjust for market conditions
        market_condition = self._determine_market_condition(market_data)
        self.market_conditions[symbol] = market_condition
        
        if market_condition == "volatile":
            config.take_profit_pct *= config.volatile_market_multiplier
            config.stop_loss_pct *= config.volatile_market_multiplier
        elif market_condition == "trending":
            config.take_profit_pct *= config.trending_market_multiplier
            config.stop_loss_pct *= config.trending_market_multiplier
        elif market_condition == "sideways":
            config.take_profit_pct *= config.sideways_market_multiplier
            config.stop_loss_pct *= config.sideways_market_multiplier
        
        # Apply limits
        config.take_profit_pct = max(config.min_tp_pct, min(config.take_profit_pct, config.max_tp_pct))
        config.stop_loss_pct = max(config.min_sl_pct, min(config.stop_loss_pct, config.max_sl_pct))
        
        return config
    
    def _determine_market_condition(self, market_data: MarketData) -> str:
        """Determine current market condition."""
        try:
            # Simple volatility-based classification
            if market_data.volatility > 0.03:  # 3% volatility threshold
                return "volatile"
            elif market_data.volatility < 0.01:  # 1% volatility threshold
                return "sideways"
            else:
                return "trending"
        except:
            return "normal"
    
    def _calculate_fixed_tpsl(self, entry_price: float, config: TPSLConfig, position_side: OrderSide) -> Tuple[float, float]:
        """Calculate fixed percentage TPSL levels."""
        if position_side == OrderSide.BUY:
            take_profit = entry_price * (1 + config.take_profit_pct)
            stop_loss = entry_price * (1 - config.stop_loss_pct)
        else:
            take_profit = entry_price * (1 - config.take_profit_pct)
            stop_loss = entry_price * (1 + config.stop_loss_pct)
        
        return take_profit, stop_loss
    
    def _calculate_atr_tpsl(self, entry_price: float, market_data: MarketData, 
                          config: TPSLConfig, position_side: OrderSide) -> Tuple[float, float]:
        """Calculate ATR-based TPSL levels."""
        try:
            # Get ATR from market data (would need historical data in real implementation)
            atr = getattr(market_data, 'atr', entry_price * 0.01)  # Default to 1% if ATR not available
            
            if position_side == OrderSide.BUY:
                take_profit = entry_price + (atr * config.atr_multiplier_tp)
                stop_loss = entry_price - (atr * config.atr_multiplier_sl)
            else:
                take_profit = entry_price - (atr * config.atr_multiplier_tp)
                stop_loss = entry_price + (atr * config.atr_multiplier_sl)
            
            return take_profit, stop_loss
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating ATR TPSL: {e}")
            return self._calculate_fixed_tpsl(entry_price, config, position_side)
    
    def _calculate_volatility_tpsl(self, entry_price: float, market_data: MarketData,
                                 config: TPSLConfig, position_side: OrderSide) -> Tuple[float, float]:
        """Calculate volatility-based TPSL levels."""
        try:
            volatility = market_data.volatility
            
            if position_side == OrderSide.BUY:
                take_profit = entry_price * (1 + volatility * config.volatility_multiplier_tp)
                stop_loss = entry_price * (1 - volatility * config.volatility_multiplier_sl)
            else:
                take_profit = entry_price * (1 - volatility * config.volatility_multiplier_tp)
                stop_loss = entry_price * (1 + volatility * config.volatility_multiplier_sl)
            
            return take_profit, stop_loss
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating volatility TPSL: {e}")
            return self._calculate_fixed_tpsl(entry_price, config, position_side)
    
    def _calculate_dynamic_tpsl(self, entry_price: float, market_data: MarketData,
                              config: TPSLConfig, position_side: OrderSide) -> Tuple[float, float]:
        """Calculate dynamic TPSL levels."""
        try:
            # Combine multiple factors for dynamic adjustment
            base_tp = config.take_profit_pct
            base_sl = config.stop_loss_pct
            
            # Adjust based on volatility
            volatility_factor = 1.0 + (market_data.volatility * config.dynamic_adjustment_factor)
            
            # Adjust based on spread
            spread_factor = 1.0 + (market_data.spread / market_data.last_price * 10)
            
            # Apply adjustments
            adjusted_tp = base_tp * volatility_factor * spread_factor
            adjusted_sl = base_sl * volatility_factor * spread_factor
            
            # Apply limits
            adjusted_tp = max(config.min_tp_pct, min(adjusted_tp, config.max_tp_pct))
            adjusted_sl = max(config.min_sl_pct, min(adjusted_sl, config.max_sl_pct))
            
            return self._calculate_fixed_tpsl(entry_price, 
                                            TPSLConfig(take_profit_pct=adjusted_tp, stop_loss_pct=adjusted_sl), 
                                            position_side)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating dynamic TPSL: {e}")
            return self._calculate_fixed_tpsl(entry_price, config, position_side)
    
    def _calculate_trailing_tpsl(self, entry_price: float, market_data: MarketData,
                               config: TPSLConfig, position_side: OrderSide) -> Tuple[float, float]:
        """Calculate trailing TPSL levels."""
        try:
            # For trailing, we need to track the highest/lowest price since entry
            # This would require position tracking in a real implementation
            
            # For now, use fixed TPSL as base
            take_profit, stop_loss = self._calculate_fixed_tpsl(entry_price, config, position_side)
            
            # In a real implementation, we would adjust these based on price movement
            # and trailing logic
            
            return take_profit, stop_loss
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating trailing TPSL: {e}")
            return self._calculate_fixed_tpsl(entry_price, config, position_side)
    
    def _calculate_scaling_tpsl(self, entry_price: float, market_data: MarketData,
                              config: TPSLConfig, position_side: OrderSide) -> Tuple[float, float]:
        """Calculate scaling TPSL levels."""
        try:
            # For scaling, we need multiple take profit levels
            # This is a simplified version - in reality, we'd track multiple levels
            
            # Use the first scale-out level as the main take profit
            first_scale_level = config.scale_out_levels[0] if config.scale_out_levels else 0.5
            
            if position_side == OrderSide.BUY:
                take_profit = entry_price * (1 + first_scale_level * config.take_profit_pct)
                stop_loss = entry_price * (1 - config.stop_loss_pct)
            else:
                take_profit = entry_price * (1 - first_scale_level * config.take_profit_pct)
                stop_loss = entry_price * (1 + config.stop_loss_pct)
            
            return take_profit, stop_loss
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating scaling TPSL: {e}")
            return self._calculate_fixed_tpsl(entry_price, config, position_side)
    
    def _calculate_time_tpsl(self, entry_price: float, market_data: MarketData,
                           config: TPSLConfig, position_side: OrderSide) -> Tuple[float, float]:
        """Calculate time-based TPSL levels."""
        try:
            # Adjust TPSL based on time decay
            # This is a simplified version - in reality, we'd track entry time
            
            time_factor = 1.0 - config.time_decay_factor  # Reduce TPSL over time
            
            adjusted_tp = config.take_profit_pct * time_factor
            adjusted_sl = config.stop_loss_pct * (1.0 + config.time_decay_factor)  # Tighten stop loss over time
            
            return self._calculate_fixed_tpsl(entry_price,
                                            TPSLConfig(take_profit_pct=adjusted_tp, stop_loss_pct=adjusted_sl),
                                            position_side)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating time TPSL: {e}")
            return self._calculate_fixed_tpsl(entry_price, config, position_side)
    
    def _calculate_momentum_tpsl(self, entry_price: float, market_data: MarketData,
                               config: TPSLConfig, position_side: OrderSide) -> Tuple[float, float]:
        """Calculate momentum-based TPSL levels."""
        try:
            # Adjust TPSL based on momentum
            # This is a simplified version - in reality, we'd calculate momentum indicators
            
            momentum_factor = 1.0  # Default factor
            
            # In a real implementation, we would:
            # 1. Calculate momentum indicators (RSI, MACD, etc.)
            # 2. Adjust TPSL based on momentum strength
            # 3. Use different TPSL for trending vs. ranging markets
            
            adjusted_tp = config.take_profit_pct * momentum_factor
            adjusted_sl = config.stop_loss_pct * momentum_factor
            
            return self._calculate_fixed_tpsl(entry_price,
                                            TPSLConfig(take_profit_pct=adjusted_tp, stop_loss_pct=adjusted_sl),
                                            position_side)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating momentum TPSL: {e}")
            return self._calculate_fixed_tpsl(entry_price, config, position_side)
    
    def _calculate_sr_tpsl(self, entry_price: float, market_data: MarketData,
                         config: TPSLConfig, position_side: OrderSide) -> Tuple[float, float]:
        """Calculate support/resistance-based TPSL levels."""
        try:
            # Adjust TPSL based on support/resistance levels
            # This is a simplified version - in reality, we'd identify S/R levels
            
            # For now, use fixed TPSL as base
            take_profit, stop_loss = self._calculate_fixed_tpsl(entry_price, config, position_side)
            
            # In a real implementation, we would:
            # 1. Identify nearby support/resistance levels
            # 2. Adjust TPSL to target S/R levels
            # 3. Use S/R levels as dynamic stop losses
            
            return take_profit, stop_loss
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating S/R TPSL: {e}")
            return self._calculate_fixed_tpsl(entry_price, config, position_side)
    
    def optimize_tpsl_parameters(self, historical_data: pd.DataFrame, 
                               symbol: str, position_side: OrderSide,
                               optimization_params: Dict[str, List[float]]) -> TPSLOptimizationResult:
        """Optimize TPSL parameters using historical data."""
        try:
            start_time = time.time()
            
            # Generate parameter combinations
            param_names = list(optimization_params.keys())
            param_values = list(optimization_params.values())
            param_combinations = list(product(*param_values))
            
            best_config = None
            best_score = -np.inf
            optimization_results = []
            
            self.logger.info(f"🔍 Optimizing TPSL parameters: {len(param_combinations)} combinations")
            
            for i, param_combo in enumerate(param_combinations):
                try:
                    # Create config with current parameters
                    config = copy.deepcopy(self.base_config)
                    for param_name, param_value in zip(param_names, param_combo):
                        setattr(config, param_name, param_value)
                    
                    # Test configuration
                    score = self._test_tpsl_config(config, historical_data, symbol, position_side)
                    
                    optimization_results.append((config, score))
                    
                    if score > best_score:
                        best_score = score
                        best_config = config
                    
                    if (i + 1) % 100 == 0:
                        self.logger.info(f"📊 Progress: {i + 1}/{len(param_combinations)} combinations tested")
                
                except Exception as e:
                    self.logger.warning(f"⚠️ Error testing parameter combination {i}: {e}")
                    continue
            
            # Calculate parameter importance
            parameter_importance = self._calculate_parameter_importance(optimization_results, param_names)
            
            optimization_time = time.time() - start_time
            
            result = TPSLOptimizationResult(
                best_config=best_config,
                best_score=best_score,
                optimization_results=optimization_results,
                parameter_importance=parameter_importance,
                optimization_time=optimization_time,
                total_tests=len(param_combinations)
            )
            
            self.logger.info(f"✅ TPSL optimization completed in {optimization_time:.2f}s")
            self.logger.info(f"📊 Best score: {best_score:.4f}")
            self.logger.info(f"📊 Best config: TP={best_config.take_profit_pct:.1%}, SL={best_config.stop_loss_pct:.1%}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing TPSL parameters: {e}")
            raise
    
    def _test_tpsl_config(self, config: TPSLConfig, historical_data: pd.DataFrame,
                         symbol: str, position_side: OrderSide) -> float:
        """Test a TPSL configuration on historical data."""
        try:
            # This is a simplified test - in reality, we'd run a full backtest
            # For now, we'll use a simple scoring function
            
            # Calculate TPSL levels for each entry point
            scores = []
            
            for i in range(len(historical_data) - 1):
                entry_price = historical_data.iloc[i]['close']
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=historical_data.iloc[i].name,
                    bid_price=entry_price * 0.9999,
                    ask_price=entry_price * 1.0001,
                    bid_size=1000,
                    ask_size=1000,
                    last_price=entry_price,
                    volume=historical_data.iloc[i]['volume'],
                    volatility=0.02,  # Default volatility
                    spread=0.0002,
                    market_condition="normal"
                )
                
                take_profit, stop_loss = self.calculate_tpsl_levels(symbol, entry_price, market_data, position_side)
                
                # Calculate potential profit/loss
                if position_side == OrderSide.BUY:
                    tp_pct = (take_profit - entry_price) / entry_price
                    sl_pct = (entry_price - stop_loss) / entry_price
                else:
                    tp_pct = (entry_price - take_profit) / entry_price
                    sl_pct = (stop_loss - entry_price) / entry_price
                
                # Simple scoring based on risk-reward ratio
                if sl_pct > 0:
                    risk_reward_ratio = tp_pct / sl_pct
                    score = risk_reward_ratio * tp_pct  # Reward higher take profits
                else:
                    score = 0
                
                scores.append(score)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error testing TPSL config: {e}")
            return 0.0
    
    def _calculate_parameter_importance(self, optimization_results: List[Tuple[TPSLConfig, float]],
                                      param_names: List[str]) -> Dict[str, float]:
        """Calculate parameter importance from optimization results."""
        try:
            importance = {}
            
            for param_name in param_names:
                # Group results by parameter value
                param_scores = {}
                
                for config, score in optimization_results:
                    param_value = getattr(config, param_name)
                    if param_value not in param_scores:
                        param_scores[param_value] = []
                    param_scores[param_value].append(score)
                
                # Calculate variance in scores for this parameter
                all_scores = [score for scores in param_scores.values() for score in scores]
                if len(all_scores) > 1:
                    importance[param_name] = np.std(all_scores)
                else:
                    importance[param_name] = 0.0
            
            # Normalize importance scores
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v / total_importance for k, v in importance.items()}
            
            return importance
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating parameter importance: {e}")
            return {param_name: 0.0 for param_name in param_names}
    
    def get_tpsl_performance_metrics(self) -> Dict[str, float]:
        """Get TPSL performance metrics."""
        try:
            if not self.tpsl_results:
                return {}
            
            # Calculate metrics
            total_trades = len(self.tpsl_results)
            winning_trades = len([r for r in self.tpsl_results if r.profit_loss > 0])
            losing_trades = len([r for r in self.tpsl_results if r.profit_loss < 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            avg_profit = np.mean([r.profit_loss for r in self.tpsl_results if r.profit_loss > 0]) if winning_trades > 0 else 0.0
            avg_loss = np.mean([r.profit_loss for r in self.tpsl_results if r.profit_loss < 0]) if losing_trades > 0 else 0.0
            
            avg_risk_reward = np.mean([r.risk_reward_ratio for r in self.tpsl_results])
            avg_hold_time = np.mean([r.hold_time_hours for r in self.tpsl_results])
            
            # TPSL effectiveness
            tp_hits = len([r for r in self.tpsl_results if r.exit_reason == "take_profit"])
            sl_hits = len([r for r in self.tpsl_results if r.exit_reason == "stop_loss"])
            
            tp_effectiveness = tp_hits / total_trades if total_trades > 0 else 0.0
            sl_effectiveness = sl_hits / total_trades if total_trades > 0 else 0.0
            
            return {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "avg_risk_reward": avg_risk_reward,
                "avg_hold_time_hours": avg_hold_time,
                "tp_effectiveness": tp_effectiveness,
                "sl_effectiveness": sl_effectiveness,
                "avg_tpsl_performance_score": np.mean([r.tpsl_performance_score for r in self.tpsl_results])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating TPSL performance metrics: {e}")
            return {}


class EnhancedABCTestingFramework(ABCTestingFramework):
    """Enhanced A/B/C testing framework with multi-model and TPSL support."""
    
    def __init__(self, config: ABCTestingConfig, tpsl_configs: Optional[Dict[str, TPSLConfig]] = None):
        """Initialize enhanced A/B/C testing framework."""
        super().__init__(config)
        
        self.logger = logger.getChild('EnhancedABCTestingFramework')
        
        # TPSL configurations for each model
        self.tpsl_configs = tpsl_configs or {}
        
        # TPSL managers for each model
        self.tpsl_managers: Dict[str, TPSLManager] = {}
        
        # Initialize TPSL managers
        self._initialize_tpsl_managers()
        
        self.logger.info("🚀 Enhanced A/B/C Testing Framework initialized")
        self.logger.info(f"📊 Models supported: {len(self.config.model_configs)}")
        self.logger.info(f"📊 TPSL configurations: {len(self.tpsl_configs)}")
    
    def _initialize_tpsl_managers(self) -> None:
        """Initialize TPSL managers for each model."""
        try:
            for model_config in self.config.model_configs:
                model_id = model_config["model_id"]
                
                # Get TPSL config for this model
                tpsl_config = self.tpsl_configs.get(model_id, TPSLConfig())
                
                # Create TPSL manager
                tpsl_manager = TPSLManager(tpsl_config)
                self.tpsl_managers[model_id] = tpsl_manager
                
                self.logger.info(f"✅ TPSL manager initialized for model: {model_id}")
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing TPSL managers: {e}")
    
    def add_model(self, model_config: Dict[str, Any], tpsl_config: Optional[TPSLConfig] = None) -> str:
        """Add a new model to the testing framework."""
        try:
            model_id = model_config["model_id"]
            
            # Add to model configs
            self.config.model_configs.append(model_config)
            
            # Add TPSL config
            if tpsl_config:
                self.tpsl_configs[model_id] = tpsl_config
            else:
                self.tpsl_configs[model_id] = TPSLConfig()
            
            # Initialize TPSL manager
            tpsl_manager = TPSLManager(self.tpsl_configs[model_id])
            self.tpsl_managers[model_id] = tpsl_manager
            
            self.logger.info(f"✅ Model added: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"❌ Error adding model: {e}")
            return ""
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the testing framework."""
        try:
            # Remove from model configs
            self.config.model_configs = [mc for mc in self.config.model_configs if mc["model_id"] != model_id]
            
            # Remove TPSL config and manager
            if model_id in self.tpsl_configs:
                del self.tpsl_configs[model_id]
            if model_id in self.tpsl_managers:
                del self.tpsl_managers[model_id]
            
            self.logger.info(f"✅ Model removed: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error removing model: {e}")
            return False
    
    def update_tpsl_config(self, model_id: str, tpsl_config: TPSLConfig) -> bool:
        """Update TPSL configuration for a model."""
        try:
            if model_id not in self.tpsl_managers:
                self.logger.error(f"❌ Model {model_id} not found")
                return False
            
            # Update TPSL config
            self.tpsl_configs[model_id] = tpsl_config
            
            # Update TPSL manager
            self.tpsl_managers[model_id] = TPSLManager(tpsl_config)
            
            self.logger.info(f"✅ TPSL config updated for model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error updating TPSL config: {e}")
            return False
    
    def optimize_tpsl_parameters(self, model_id: str, historical_data: pd.DataFrame,
                               optimization_params: Dict[str, List[float]]) -> Optional[TPSLOptimizationResult]:
        """Optimize TPSL parameters for a specific model."""
        try:
            if model_id not in self.tpsl_managers:
                self.logger.error(f"❌ Model {model_id} not found")
                return None
            
            tpsl_manager = self.tpsl_managers[model_id]
            
            # Run optimization
            result = tpsl_manager.optimize_tpsl_parameters(
                historical_data=historical_data,
                symbol=self.config.symbol,
                position_side=OrderSide.BUY,  # Default to buy side
                optimization_params=optimization_params
            )
            
            # Update TPSL config with optimized parameters
            self.update_tpsl_config(model_id, result.best_config)
            
            self.logger.info(f"✅ TPSL parameters optimized for model: {model_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing TPSL parameters: {e}")
            return None
    
    def get_tpsl_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get TPSL performance summary for all models."""
        try:
            summary = {}
            
            for model_id, tpsl_manager in self.tpsl_managers.items():
                summary[model_id] = tpsl_manager.get_tpsl_performance_metrics()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error getting TPSL performance summary: {e}")
            return {}
    
    def create_tpsl_parameter_grid(self, base_params: Dict[str, Any]) -> Dict[str, List[float]]:
        """Create a parameter grid for TPSL optimization."""
        try:
            # Default parameter ranges
            default_ranges = {
                "take_profit_pct": [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
                "stop_loss_pct": [0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
                "atr_multiplier_tp": [1.5, 2.0, 2.5, 3.0],
                "atr_multiplier_sl": [0.5, 1.0, 1.5, 2.0],
                "volatility_multiplier_tp": [1.0, 1.5, 2.0, 2.5],
                "volatility_multiplier_sl": [0.5, 1.0, 1.5, 2.0]
            }
            
            # Use provided parameters or defaults
            param_grid = {}
            for param_name, param_values in base_params.items():
                if param_name in default_ranges:
                    param_grid[param_name] = param_values
                else:
                    param_grid[param_name] = default_ranges.get(param_name, [0.02])
            
            return param_grid
            
        except Exception as e:
            self.logger.error(f"❌ Error creating TPSL parameter grid: {e}")
            return {"take_profit_pct": [0.02], "stop_loss_pct": [0.01]}


# Convenience function for easy integration
def create_enhanced_abc_framework(config: ABCTestingConfig, 
                                tpsl_configs: Optional[Dict[str, TPSLConfig]] = None) -> EnhancedABCTestingFramework:
    """Create an enhanced A/B/C testing framework instance."""
    return EnhancedABCTestingFramework(config, tpsl_configs)