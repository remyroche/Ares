"""
Risk Management and Position Sizing System

This module provides comprehensive risk management and position sizing
capabilities for A/B/C testing of multiple models.

Key Features:
- Dynamic position sizing based on volatility and risk metrics
- Portfolio-level risk controls and limits
- Correlation-based risk management
- Drawdown protection and circuit breakers
- Real-time risk monitoring and alerts
- Advanced position sizing algorithms (Kelly, Fixed Fractional, etc.)
- Risk-adjusted performance metrics
- Stress testing and scenario analysis
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
from collections import defaultdict, deque
import scipy.stats as stats
from scipy.optimize import minimize

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)

# Core decorators and validation
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


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PositionSizingMethod(Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY = "kelly"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"
    OPTIMAL_F = "optimal_f"
    ATR_BASED = "atr_based"
    CORRELATION_ADJUSTED = "correlation_adjusted"


class RiskMetric(Enum):
    """Risk metrics."""
    VAR = "var"  # Value at Risk
    CVAR = "cvar"  # Conditional Value at Risk
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    CORRELATION = "correlation"


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_portfolio_risk: float = 0.20  # 20% max portfolio risk
    max_position_risk: float = 0.05  # 5% max position risk
    max_correlation: float = 0.70  # 70% max correlation
    max_drawdown: float = 0.15  # 15% max drawdown
    max_leverage: float = 1.0  # No leverage
    max_concurrent_positions: int = 10
    max_daily_loss: float = 0.05  # 5% max daily loss
    var_confidence_level: float = 0.95  # 95% VaR
    cvar_confidence_level: float = 0.95  # 95% CVaR
    enable_circuit_breakers: bool = True
    circuit_breaker_threshold: float = 0.10  # 10% loss triggers circuit breaker


@dataclass
class PositionSizingConfig:
    """Position sizing configuration."""
    method: PositionSizingMethod = PositionSizingMethod.FIXED_FRACTIONAL
    base_risk_per_trade: float = 0.02  # 2% base risk
    max_position_size: float = 0.10  # 10% max position
    min_position_size: float = 0.001  # 0.1% min position
    volatility_lookback: int = 20  # 20 periods for volatility
    correlation_lookback: int = 60  # 60 periods for correlation
    kelly_fraction: float = 0.25  # 25% of Kelly optimal
    atr_multiplier: float = 2.0  # 2x ATR for stop loss
    enable_dynamic_sizing: bool = True
    enable_correlation_adjustment: bool = True


@dataclass
class RiskMetrics:
    """Risk metrics container."""
    portfolio_value: float
    portfolio_var: float
    portfolio_cvar: float
    portfolio_volatility: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    correlation_matrix: np.ndarray
    position_risks: Dict[str, float]
    total_exposure: float
    leverage: float
    risk_level: RiskLevel
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PositionSizingResult:
    """Position sizing calculation result."""
    symbol: str
    recommended_size: float
    risk_amount: float
    position_value: float
    stop_loss_price: float
    risk_reward_ratio: float
    confidence_score: float
    sizing_method: PositionSizingMethod
    risk_factors: Dict[str, float]
    warnings: List[str] = field(default_factory=list)


class RiskCalculator:
    """Advanced risk calculation engine."""
    
    def __init__(self, risk_limits: RiskLimits):
        """Initialize risk calculator."""
        self.risk_limits = risk_limits
        self.logger = logger.getChild('RiskCalculator')
        
        # Historical data for calculations
        self.returns_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        self.portfolio_history: deque = deque(maxlen=252)
        
        self.logger.info("🚀 RiskCalculator initialized")
        self.logger.info(f"📊 Max portfolio risk: {risk_limits.max_portfolio_risk:.1%}")
        self.logger.info(f"📊 Max position risk: {risk_limits.max_position_risk:.1%}")
    
    def calculate_portfolio_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR) for portfolio."""
        try:
            if len(returns) < 30:
                return 0.0
            
            # Historical simulation method
            var_percentile = (1 - confidence_level) * 100
            var = np.percentile(returns, var_percentile)
            
            return abs(var)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating VaR: {e}")
            return 0.0
    
    def calculate_portfolio_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR) for portfolio."""
        try:
            if len(returns) < 30:
                return 0.0
            
            # Calculate VaR first
            var = self.calculate_portfolio_var(returns, confidence_level)
            
            # CVaR is the mean of returns below VaR
            tail_returns = returns[returns <= -var]
            cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var
            
            return abs(cvar)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating CVaR: {e}")
            return 0.0
    
    def calculate_portfolio_volatility(self, returns: np.ndarray, annualized: bool = True) -> float:
        """Calculate portfolio volatility."""
        try:
            if len(returns) < 2:
                return 0.0
            
            volatility = np.std(returns)
            
            if annualized:
                volatility *= np.sqrt(252)  # Annualize
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating volatility: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            if len(portfolio_values) < 2:
                return 0.0
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(portfolio_values)
            
            # Calculate drawdowns
            drawdowns = (portfolio_values - running_max) / running_max
            
            # Return maximum drawdown
            return abs(np.min(drawdowns))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating max drawdown: {e}")
            return 0.0
    
    def calculate_correlation_matrix(self, returns_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate correlation matrix for assets."""
        try:
            if len(returns_data) < 2:
                return np.array([[1.0]])
            
            # Align returns data
            symbols = list(returns_data.keys())
            min_length = min(len(returns) for returns in returns_data.values())
            
            if min_length < 10:
                return np.eye(len(symbols))
            
            # Create aligned returns matrix
            aligned_returns = np.array([
                returns_data[symbol][-min_length:] for symbol in symbols
            ])
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(aligned_returns)
            
            # Handle NaN values
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating correlation matrix: {e}")
            return np.eye(len(returns_data))
    
    def calculate_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Calculate beta coefficient."""
        try:
            if len(asset_returns) < 30 or len(market_returns) < 30:
                return 1.0
            
            # Align returns
            min_length = min(len(asset_returns), len(market_returns))
            asset_returns = asset_returns[-min_length:]
            market_returns = market_returns[-min_length:]
            
            # Calculate covariance and variance
            covariance = np.cov(asset_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            return beta
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating beta: {e}")
            return 1.0
    
    def calculate_risk_metrics(self, portfolio_value: float, 
                             positions: Dict[str, Dict[str, float]],
                             market_data: Dict[str, Dict[str, float]]) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        try:
            # Update history
            self.portfolio_history.append(portfolio_value)
            
            # Calculate portfolio returns
            if len(self.portfolio_history) > 1:
                portfolio_returns = np.diff(self.portfolio_history)
            else:
                portfolio_returns = np.array([0.0])
            
            # Calculate risk metrics
            var = self.calculate_portfolio_var(portfolio_returns, self.risk_limits.var_confidence_level)
            cvar = self.calculate_portfolio_cvar(portfolio_returns, self.risk_limits.var_confidence_level)
            volatility = self.calculate_portfolio_volatility(portfolio_returns)
            max_drawdown = self.calculate_max_drawdown(np.array(self.portfolio_history))
            
            # Current drawdown
            if len(self.portfolio_history) > 1:
                peak = np.max(self.portfolio_history)
                current_drawdown = (peak - portfolio_value) / peak
            else:
                current_drawdown = 0.0
            
            # Calculate Sharpe ratio
            if volatility > 0:
                sharpe_ratio = np.mean(portfolio_returns) * np.sqrt(252) / volatility
            else:
                sharpe_ratio = 0.0
            
            # Calculate Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            if len(downside_returns) > 0:
                downside_volatility = np.std(downside_returns) * np.sqrt(252)
                sortino_ratio = np.mean(portfolio_returns) * np.sqrt(252) / downside_volatility
            else:
                sortino_ratio = 0.0
            
            # Calculate Calmar ratio
            if max_drawdown > 0:
                annual_return = np.mean(portfolio_returns) * 252
                calmar_ratio = annual_return / max_drawdown
            else:
                calmar_ratio = 0.0
            
            # Calculate position risks
            position_risks = {}
            total_exposure = 0.0
            
            for symbol, position in positions.items():
                if symbol in market_data:
                    position_value = position.get('quantity', 0) * market_data[symbol].get('price', 0)
                    position_risk = position_value / portfolio_value
                    position_risks[symbol] = position_risk
                    total_exposure += position_value
            
            # Calculate leverage
            leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0.0
            
            # Calculate correlation matrix
            returns_data = {}
            for symbol in positions.keys():
                if symbol in self.returns_history and len(self.returns_history[symbol]) > 10:
                    returns_data[symbol] = np.array(self.returns_history[symbol])
            
            correlation_matrix = self.calculate_correlation_matrix(returns_data)
            
            # Determine risk level
            risk_level = self._determine_risk_level(
                current_drawdown, max_drawdown, leverage, var, volatility
            )
            
            return RiskMetrics(
                portfolio_value=portfolio_value,
                portfolio_var=var,
                portfolio_cvar=cvar,
                portfolio_volatility=volatility,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                beta=1.0,  # Would need market benchmark
                correlation_matrix=correlation_matrix,
                position_risks=position_risks,
                total_exposure=total_exposure,
                leverage=leverage,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk metrics: {e}")
            return RiskMetrics(
                portfolio_value=portfolio_value,
                portfolio_var=0.0,
                portfolio_cvar=0.0,
                portfolio_volatility=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                beta=1.0,
                correlation_matrix=np.array([[1.0]]),
                position_risks={},
                total_exposure=0.0,
                leverage=0.0,
                risk_level=RiskLevel.LOW
            )
    
    def _determine_risk_level(self, current_drawdown: float, max_drawdown: float,
                            leverage: float, var: float, volatility: float) -> RiskLevel:
        """Determine current risk level."""
        # Check critical conditions
        if (current_drawdown > self.risk_limits.max_drawdown or
            leverage > self.risk_limits.max_leverage or
            var > self.risk_limits.max_portfolio_risk):
            return RiskLevel.CRITICAL
        
        # Check high risk conditions
        if (current_drawdown > self.risk_limits.max_drawdown * 0.8 or
            leverage > self.risk_limits.max_leverage * 0.8 or
            var > self.risk_limits.max_portfolio_risk * 0.8):
            return RiskLevel.HIGH
        
        # Check medium risk conditions
        if (current_drawdown > self.risk_limits.max_drawdown * 0.5 or
            leverage > self.risk_limits.max_leverage * 0.5 or
            var > self.risk_limits.max_portfolio_risk * 0.5):
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def update_returns_history(self, symbol: str, returns: float) -> None:
        """Update returns history for a symbol."""
        self.returns_history[symbol].append(returns)
    
    def update_price_history(self, symbol: str, price: float) -> None:
        """Update price history for a symbol."""
        self.price_history[symbol].append(price)


class PositionSizer:
    """Advanced position sizing engine."""
    
    def __init__(self, config: PositionSizingConfig, risk_calculator: RiskCalculator):
        """Initialize position sizer."""
        self.config = config
        self.risk_calculator = risk_calculator
        self.logger = logger.getChild('PositionSizer')
        
        self.logger.info("🚀 PositionSizer initialized")
        self.logger.info(f"📊 Sizing method: {config.method.value}")
        self.logger.info(f"📊 Base risk per trade: {config.base_risk_per_trade:.1%}")
    
    def calculate_position_size(self, symbol: str, entry_price: float,
                              stop_loss_price: float, portfolio_value: float,
                              current_positions: Dict[str, Dict[str, float]],
                              market_data: Dict[str, Dict[str, float]]) -> PositionSizingResult:
        """Calculate optimal position size."""
        try:
            # Calculate base risk amount
            risk_amount = portfolio_value * self.config.base_risk_per_trade
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss_price)
            
            if risk_per_share <= 0:
                return self._create_error_result(symbol, "Invalid stop loss price")
            
            # Calculate base position size
            base_size = risk_amount / risk_per_share
            
            # Apply position sizing method
            if self.config.method == PositionSizingMethod.FIXED:
                recommended_size = self._fixed_sizing(base_size, portfolio_value)
            elif self.config.method == PositionSizingMethod.FIXED_FRACTIONAL:
                recommended_size = self._fixed_fractional_sizing(base_size, portfolio_value)
            elif self.config.method == PositionSizingMethod.KELLY:
                recommended_size = self._kelly_sizing(symbol, base_size, portfolio_value)
            elif self.config.method == PositionSizingMethod.VOLATILITY_ADJUSTED:
                recommended_size = self._volatility_adjusted_sizing(symbol, base_size, portfolio_value)
            elif self.config.method == PositionSizingMethod.RISK_PARITY:
                recommended_size = self._risk_parity_sizing(symbol, base_size, portfolio_value, current_positions)
            elif self.config.method == PositionSizingMethod.OPTIMAL_F:
                recommended_size = self._optimal_f_sizing(symbol, base_size, portfolio_value)
            elif self.config.method == PositionSizingMethod.ATR_BASED:
                recommended_size = self._atr_based_sizing(symbol, base_size, portfolio_value, market_data)
            elif self.config.method == PositionSizingMethod.CORRELATION_ADJUSTED:
                recommended_size = self._correlation_adjusted_sizing(symbol, base_size, portfolio_value, current_positions)
            else:
                recommended_size = self._fixed_fractional_sizing(base_size, portfolio_value)
            
            # Apply limits
            recommended_size = self._apply_limits(recommended_size, portfolio_value, entry_price)
            
            # Calculate final metrics
            position_value = recommended_size * entry_price
            risk_reward_ratio = self._calculate_risk_reward_ratio(entry_price, stop_loss_price, market_data.get(symbol, {}))
            confidence_score = self._calculate_confidence_score(symbol, market_data)
            
            # Generate warnings
            warnings = self._generate_warnings(recommended_size, position_value, portfolio_value, current_positions)
            
            return PositionSizingResult(
                symbol=symbol,
                recommended_size=recommended_size,
                risk_amount=risk_amount,
                position_value=position_value,
                stop_loss_price=stop_loss_price,
                risk_reward_ratio=risk_reward_ratio,
                confidence_score=confidence_score,
                sizing_method=self.config.method,
                risk_factors=self._calculate_risk_factors(symbol, market_data),
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return self._create_error_result(symbol, str(e))
    
    def _fixed_sizing(self, base_size: float, portfolio_value: float) -> float:
        """Fixed position sizing."""
        return base_size
    
    def _fixed_fractional_sizing(self, base_size: float, portfolio_value: float) -> float:
        """Fixed fractional position sizing."""
        max_position_value = portfolio_value * self.config.max_position_size
        max_size = max_position_value / (base_size * 100)  # Assuming $100 per share
        return min(base_size, max_size)
    
    def _kelly_sizing(self, symbol: str, base_size: float, portfolio_value: float) -> float:
        """Kelly criterion position sizing."""
        try:
            if symbol not in self.risk_calculator.returns_history:
                return base_size
            
            returns = np.array(self.risk_calculator.returns_history[symbol])
            if len(returns) < 30:
                return base_size
            
            # Calculate win rate and average win/loss
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            
            if len(wins) == 0 or len(losses) == 0:
                return base_size
            
            win_rate = len(wins) / len(returns)
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            
            if avg_loss == 0:
                return base_size
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply Kelly fraction and cap at maximum
            kelly_size = base_size * kelly_fraction * self.config.kelly_fraction
            max_size = portfolio_value * self.config.max_position_size / 100  # Assuming $100 per share
            
            return min(kelly_size, max_size)
            
        except Exception as e:
            self.logger.error(f"❌ Error in Kelly sizing: {e}")
            return base_size
    
    def _volatility_adjusted_sizing(self, symbol: str, base_size: float, portfolio_value: float) -> float:
        """Volatility-adjusted position sizing."""
        try:
            if symbol not in self.risk_calculator.returns_history:
                return base_size
            
            returns = np.array(self.risk_calculator.returns_history[symbol])
            if len(returns) < self.config.volatility_lookback:
                return base_size
            
            # Calculate volatility
            volatility = np.std(returns[-self.config.volatility_lookback:])
            
            # Adjust size inversely to volatility
            volatility_adjustment = 1.0 / (1.0 + volatility * 10)  # Scale factor
            
            return base_size * volatility_adjustment
            
        except Exception as e:
            self.logger.error(f"❌ Error in volatility-adjusted sizing: {e}")
            return base_size
    
    def _risk_parity_sizing(self, symbol: str, base_size: float, portfolio_value: float,
                          current_positions: Dict[str, Dict[str, float]]) -> float:
        """Risk parity position sizing."""
        try:
            # Calculate target risk per position
            num_positions = len(current_positions) + 1  # Including new position
            target_risk_per_position = 1.0 / num_positions
            
            # Calculate current portfolio risk
            total_risk = sum(pos.get('risk', 0) for pos in current_positions.values())
            
            # Calculate target risk for new position
            target_risk = target_risk_per_position - (total_risk / num_positions)
            
            if target_risk <= 0:
                return 0.0
            
            # Convert risk to position size
            risk_per_share = 0.02  # Assume 2% risk per share
            return target_risk * portfolio_value / risk_per_share
            
        except Exception as e:
            self.logger.error(f"❌ Error in risk parity sizing: {e}")
            return base_size
    
    def _optimal_f_sizing(self, symbol: str, base_size: float, portfolio_value: float) -> float:
        """Optimal f position sizing."""
        try:
            if symbol not in self.risk_calculator.returns_history:
                return base_size
            
            returns = np.array(self.risk_calculator.returns_history[symbol])
            if len(returns) < 30:
                return base_size
            
            # Calculate optimal f using geometric mean
            def geometric_mean(f):
                if f <= 0 or f >= 1:
                    return -np.inf
                
                # Calculate HPR (Holding Period Return) for each trade
                hprs = 1 + f * returns
                hprs = hprs[hprs > 0]  # Remove negative HPRs
                
                if len(hprs) == 0:
                    return -np.inf
                
                # Calculate geometric mean
                return np.prod(hprs) ** (1.0 / len(hprs))
            
            # Find optimal f
            result = minimize(lambda f: -geometric_mean(f), x0=0.1, bounds=[(0.001, 0.5)])
            
            if result.success:
                optimal_f = result.x[0]
                return base_size * optimal_f
            else:
                return base_size
            
        except Exception as e:
            self.logger.error(f"❌ Error in optimal f sizing: {e}")
            return base_size
    
    def _atr_based_sizing(self, symbol: str, base_size: float, portfolio_value: float,
                         market_data: Dict[str, Dict[str, float]]) -> float:
        """ATR-based position sizing."""
        try:
            if symbol not in market_data or 'atr' not in market_data[symbol]:
                return base_size
            
            atr = market_data[symbol]['atr']
            if atr <= 0:
                return base_size
            
            # Calculate position size based on ATR
            risk_amount = portfolio_value * self.config.base_risk_per_trade
            stop_distance = atr * self.config.atr_multiplier
            
            atr_size = risk_amount / stop_distance
            
            return min(atr_size, base_size)
            
        except Exception as e:
            self.logger.error(f"❌ Error in ATR-based sizing: {e}")
            return base_size
    
    def _correlation_adjusted_sizing(self, symbol: str, base_size: float, portfolio_value: float,
                                   current_positions: Dict[str, Dict[str, float]]) -> float:
        """Correlation-adjusted position sizing."""
        try:
            if not current_positions:
                return base_size
            
            # Calculate correlation with existing positions
            total_correlation = 0.0
            correlation_count = 0
            
            for existing_symbol in current_positions.keys():
                if (existing_symbol in self.risk_calculator.returns_history and
                    symbol in self.risk_calculator.returns_history):
                    
                    existing_returns = np.array(self.risk_calculator.returns_history[existing_symbol])
                    symbol_returns = np.array(self.risk_calculator.returns_history[symbol])
                    
                    if len(existing_returns) > 10 and len(symbol_returns) > 10:
                        min_length = min(len(existing_returns), len(symbol_returns))
                        correlation = np.corrcoef(
                            existing_returns[-min_length:],
                            symbol_returns[-min_length:]
                        )[0, 1]
                        
                        if not np.isnan(correlation):
                            total_correlation += abs(correlation)
                            correlation_count += 1
            
            if correlation_count == 0:
                return base_size
            
            # Calculate average correlation
            avg_correlation = total_correlation / correlation_count
            
            # Adjust size based on correlation
            correlation_adjustment = 1.0 - (avg_correlation * 0.5)  # Reduce size by 50% of correlation
            
            return base_size * max(correlation_adjustment, 0.1)  # Minimum 10% of original size
            
        except Exception as e:
            self.logger.error(f"❌ Error in correlation-adjusted sizing: {e}")
            return base_size
    
    def _apply_limits(self, size: float, portfolio_value: float, price: float) -> float:
        """Apply position size limits."""
        # Minimum size
        size = max(size, self.config.min_position_size)
        
        # Maximum position value
        max_position_value = portfolio_value * self.config.max_position_size
        max_size = max_position_value / price if price > 0 else size
        
        return min(size, max_size)
    
    def _calculate_risk_reward_ratio(self, entry_price: float, stop_loss_price: float,
                                   market_data: Dict[str, float]) -> float:
        """Calculate risk-reward ratio."""
        try:
            risk = abs(entry_price - stop_loss_price)
            
            # Estimate target price (could be based on technical analysis)
            target_price = entry_price * 1.02  # Assume 2% target
            
            reward = abs(target_price - entry_price)
            
            return reward / risk if risk > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk-reward ratio: {e}")
            return 0.0
    
    def _calculate_confidence_score(self, symbol: str, market_data: Dict[str, Dict[str, float]]) -> float:
        """Calculate confidence score for position sizing."""
        try:
            score = 0.5  # Base score
            
            # Historical data availability
            if symbol in self.risk_calculator.returns_history:
                returns_count = len(self.risk_calculator.returns_history[symbol])
                if returns_count > 100:
                    score += 0.2
                elif returns_count > 50:
                    score += 0.1
            
            # Market data quality
            if symbol in market_data:
                if 'volume' in market_data[symbol] and market_data[symbol]['volume'] > 0:
                    score += 0.1
                if 'volatility' in market_data[symbol]:
                    score += 0.1
                if 'atr' in market_data[symbol]:
                    score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating confidence score: {e}")
            return 0.5
    
    def _calculate_risk_factors(self, symbol: str, market_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate risk factors for the position."""
        risk_factors = {}
        
        try:
            # Volatility risk
            if symbol in market_data and 'volatility' in market_data[symbol]:
                risk_factors['volatility'] = market_data[symbol]['volatility']
            
            # Liquidity risk
            if symbol in market_data and 'volume' in market_data[symbol]:
                volume = market_data[symbol]['volume']
                risk_factors['liquidity'] = 1.0 / (1.0 + volume / 1000000)  # Lower volume = higher risk
            
            # Market condition risk
            if symbol in market_data and 'market_condition' in market_data[symbol]:
                condition = market_data[symbol]['market_condition']
                condition_risk = {
                    'normal': 0.1,
                    'volatile': 0.3,
                    'illiquid': 0.5,
                    'halted': 1.0,
                    'gapping': 0.7
                }
                risk_factors['market_condition'] = condition_risk.get(condition, 0.1)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk factors: {e}")
        
        return risk_factors
    
    def _generate_warnings(self, size: float, position_value: float, portfolio_value: float,
                         current_positions: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate warnings for position sizing."""
        warnings = []
        
        try:
            # Position size warnings
            if size < self.config.min_position_size:
                warnings.append(f"Position size below minimum: {size:.4f} < {self.config.min_position_size:.4f}")
            
            if position_value > portfolio_value * self.config.max_position_size:
                warnings.append(f"Position value exceeds maximum: {position_value:.2f} > {portfolio_value * self.config.max_position_size:.2f}")
            
            # Portfolio concentration warnings
            total_exposure = sum(pos.get('value', 0) for pos in current_positions.values()) + position_value
            if total_exposure > portfolio_value * 0.8:
                warnings.append(f"High portfolio concentration: {total_exposure/portfolio_value:.1%}")
            
            # Number of positions warning
            if len(current_positions) >= 10:
                warnings.append(f"High number of positions: {len(current_positions) + 1}")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating warnings: {e}")
        
        return warnings
    
    def _create_error_result(self, symbol: str, error_message: str) -> PositionSizingResult:
        """Create error result."""
        return PositionSizingResult(
            symbol=symbol,
            recommended_size=0.0,
            risk_amount=0.0,
            position_value=0.0,
            stop_loss_price=0.0,
            risk_reward_ratio=0.0,
            confidence_score=0.0,
            sizing_method=self.config.method,
            risk_factors={},
            warnings=[error_message]
        )


class RiskManager:
    """Comprehensive risk management system."""
    
    def __init__(self, risk_limits: RiskLimits, position_sizing_config: PositionSizingConfig):
        """Initialize risk manager."""
        self.risk_limits = risk_limits
        self.position_sizing_config = position_sizing_config
        
        self.logger = logger.getChild('RiskManager')
        
        # Core components
        self.risk_calculator = RiskCalculator(risk_limits)
        self.position_sizer = PositionSizer(position_sizing_config, self.risk_calculator)
        
        # Risk monitoring
        self.risk_alerts: List[Dict[str, Any]] = []
        self.circuit_breaker_active: bool = False
        self.circuit_breaker_triggered_at: Optional[datetime] = None
        
        self.logger.info("🚀 RiskManager initialized")
        self.logger.info(f"📊 Risk limits: {risk_limits.max_portfolio_risk:.1%} portfolio, {risk_limits.max_position_risk:.1%} position")
        self.logger.info(f"📊 Circuit breakers: {'enabled' if risk_limits.enable_circuit_breakers else 'disabled'}")
    
    def check_risk_limits(self, portfolio_value: float, positions: Dict[str, Dict[str, float]],
                         market_data: Dict[str, Dict[str, float]]) -> Tuple[bool, List[str]]:
        """Check if portfolio is within risk limits."""
        try:
            violations = []
            
            # Calculate risk metrics
            risk_metrics = self.risk_calculator.calculate_risk_metrics(
                portfolio_value, positions, market_data
            )
            
            # Check portfolio risk limits
            if risk_metrics.portfolio_var > self.risk_limits.max_portfolio_risk:
                violations.append(f"Portfolio VaR exceeds limit: {risk_metrics.portfolio_var:.1%} > {self.risk_limits.max_portfolio_risk:.1%}")
            
            if risk_metrics.max_drawdown > self.risk_limits.max_drawdown:
                violations.append(f"Max drawdown exceeds limit: {risk_metrics.max_drawdown:.1%} > {self.risk_limits.max_drawdown:.1%}")
            
            if risk_metrics.leverage > self.risk_limits.max_leverage:
                violations.append(f"Leverage exceeds limit: {risk_metrics.leverage:.1%} > {self.risk_limits.max_leverage:.1%}")
            
            # Check position risk limits
            for symbol, position_risk in risk_metrics.position_risks.items():
                if position_risk > self.risk_limits.max_position_risk:
                    violations.append(f"Position {symbol} risk exceeds limit: {position_risk:.1%} > {self.risk_limits.max_position_risk:.1%}")
            
            # Check correlation limits
            if risk_metrics.correlation_matrix.size > 1:
                max_correlation = np.max(np.abs(risk_metrics.correlation_matrix - np.eye(risk_metrics.correlation_matrix.shape[0])))
                if max_correlation > self.risk_limits.max_correlation:
                    violations.append(f"Max correlation exceeds limit: {max_correlation:.1%} > {self.risk_limits.max_correlation:.1%}")
            
            # Check number of positions
            if len(positions) > self.risk_limits.max_concurrent_positions:
                violations.append(f"Too many positions: {len(positions)} > {self.risk_limits.max_concurrent_positions}")
            
            # Check circuit breaker
            if self.risk_limits.enable_circuit_breakers:
                if risk_metrics.current_drawdown > self.risk_limits.circuit_breaker_threshold:
                    if not self.circuit_breaker_active:
                        self._trigger_circuit_breaker(risk_metrics.current_drawdown)
                    violations.append(f"Circuit breaker triggered: {risk_metrics.current_drawdown:.1%} drawdown")
            
            return len(violations) == 0, violations
            
        except Exception as e:
            self.logger.error(f"❌ Error checking risk limits: {e}")
            return False, [str(e)]
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss_price: float,
                              portfolio_value: float, current_positions: Dict[str, Dict[str, float]],
                              market_data: Dict[str, Dict[str, float]]) -> PositionSizingResult:
        """Calculate optimal position size with risk management."""
        try:
            # Check if circuit breaker is active
            if self.circuit_breaker_active:
                return self.position_sizer._create_error_result(symbol, "Circuit breaker is active")
            
            # Check risk limits before sizing
            is_within_limits, violations = self.check_risk_limits(
                portfolio_value, current_positions, market_data
            )
            
            if not is_within_limits:
                return self.position_sizer._create_error_result(symbol, f"Risk limits violated: {', '.join(violations)}")
            
            # Calculate position size
            result = self.position_sizer.calculate_position_size(
                symbol, entry_price, stop_loss_price, portfolio_value,
                current_positions, market_data
            )
            
            # Add risk management warnings
            if violations:
                result.warnings.extend([f"Risk warning: {v}" for v in violations])
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return self.position_sizer._create_error_result(symbol, str(e))
    
    def _trigger_circuit_breaker(self, drawdown: float) -> None:
        """Trigger circuit breaker."""
        self.circuit_breaker_active = True
        self.circuit_breaker_triggered_at = datetime.now()
        
        alert = {
            "type": "circuit_breaker",
            "message": f"Circuit breaker triggered at {drawdown:.1%} drawdown",
            "timestamp": datetime.now(),
            "severity": "critical"
        }
        
        self.risk_alerts.append(alert)
        self.logger.critical(f"🚨 Circuit breaker triggered: {drawdown:.1%} drawdown")
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker."""
        self.circuit_breaker_active = False
        self.circuit_breaker_triggered_at = None
        
        alert = {
            "type": "circuit_breaker_reset",
            "message": "Circuit breaker reset",
            "timestamp": datetime.now(),
            "severity": "info"
        }
        
        self.risk_alerts.append(alert)
        self.logger.info("✅ Circuit breaker reset")
    
    def get_risk_summary(self, portfolio_value: float, positions: Dict[str, Dict[str, float]],
                        market_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        try:
            # Calculate risk metrics
            risk_metrics = self.risk_calculator.calculate_risk_metrics(
                portfolio_value, positions, market_data
            )
            
            # Check risk limits
            is_within_limits, violations = self.check_risk_limits(
                portfolio_value, positions, market_data
            )
            
            return {
                "risk_metrics": {
                    "portfolio_var": risk_metrics.portfolio_var,
                    "portfolio_cvar": risk_metrics.portfolio_cvar,
                    "portfolio_volatility": risk_metrics.portfolio_volatility,
                    "max_drawdown": risk_metrics.max_drawdown,
                    "current_drawdown": risk_metrics.current_drawdown,
                    "sharpe_ratio": risk_metrics.sharpe_ratio,
                    "sortino_ratio": risk_metrics.sortino_ratio,
                    "calmar_ratio": risk_metrics.calmar_ratio,
                    "leverage": risk_metrics.leverage,
                    "risk_level": risk_metrics.risk_level.value
                },
                "risk_limits": {
                    "max_portfolio_risk": self.risk_limits.max_portfolio_risk,
                    "max_position_risk": self.risk_limits.max_position_risk,
                    "max_drawdown": self.risk_limits.max_drawdown,
                    "max_leverage": self.risk_limits.max_leverage,
                    "max_correlation": self.risk_limits.max_correlation,
                    "max_concurrent_positions": self.risk_limits.max_concurrent_positions
                },
                "compliance": {
                    "within_limits": is_within_limits,
                    "violations": violations,
                    "circuit_breaker_active": self.circuit_breaker_active
                },
                "position_risks": risk_metrics.position_risks,
                "alerts": self.risk_alerts[-10:],  # Last 10 alerts
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting risk summary: {e}")
            return {"error": str(e)}
    
    def update_market_data(self, symbol: str, price: float, returns: float) -> None:
        """Update market data for risk calculations."""
        self.risk_calculator.update_price_history(symbol, price)
        self.risk_calculator.update_returns_history(symbol, returns)


# Convenience function for easy integration
def create_risk_manager(risk_limits: RiskLimits, 
                       position_sizing_config: PositionSizingConfig) -> RiskManager:
    """Create a risk manager instance."""
    return RiskManager(risk_limits, position_sizing_config)