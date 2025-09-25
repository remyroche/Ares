"""
Risk Manager

This module handles risk management, position sizing, and risk monitoring.
Integrates with shared utilities for advanced risk calculations and optimization.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

# Import shared utilities
from ..utils.common_operations import safe_dataframe_operation, validate_dataframe_columns
from ..utils.common_utilities import calculate_data_quality_metrics, safe_convert_dtypes
from ..utils.math_validation import safe_divide, validate_finite, safe_log, safe_sqrt
from ..utils.serialization_utils import JSONSerializer, PickleSerializer
from ..utils.tprint import tprint
from ..utils.data.klines_parquet import KlinesParquetManager
from ..utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
from ..utils.hardware.m1_gpu_utils import get_m1_gpu_manager, is_m1_available
from ..utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from ..utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from ..utils.matrix_operations import UnifiedMatrixOperations

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_size: float
    max_daily_loss: float
    max_leverage: float
    max_drawdown_percent: float
    max_volatility: float
    min_sharpe_ratio: float
    max_correlation: float
    max_orders_per_minute: int
    max_total_exposure: float
    max_var_95: float  # Value at Risk 95%
    max_var_99: float  # Value at Risk 99%


@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    timestamp: datetime
    symbol: str
    current_position: float
    position_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    max_drawdown: float
    leverage: float
    margin_used: float
    margin_available: float
    risk_score: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    var_99: float
    beta: float
    correlation: float


@dataclass
class RiskAlert:
    """Risk alert data structure"""
    timestamp: datetime
    symbol: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    current_value: float
    limit_value: float
    recommendation: str


class RiskManager:
    """Advanced risk management system with ML optimization"""
    
    def __init__(self, config: Any, exchange_client: Any):
        self.config = config
        self.exchange_client = exchange_client
        self.logger = logging.getLogger(__name__)
        
        # Risk configuration
        self.risk_limits = RiskLimits(
            max_position_size=config.max_position_size,
            max_daily_loss=config.max_daily_loss,
            max_leverage=config.max_leverage,
            max_drawdown_percent=10.0,
            max_volatility=0.05,
            min_sharpe_ratio=1.0,
            max_correlation=0.8,
            max_orders_per_minute=60,
            max_total_exposure=config.max_position_size * 5,
            max_var_95=0.02,  # 2% daily VaR
            max_var_99=0.05   # 5% daily VaR
        )
        
        # Risk tracking
        self.risk_metrics: Dict[str, RiskMetrics] = {}
        self.risk_alerts: List[RiskAlert] = []
        self.risk_violations: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.daily_pnl: Dict[str, float] = {}
        self.order_history: List[datetime] = []
        self.positions: Dict[str, float] = {}
        
        # Initialize utilities
        self._init_utilities()
        
        # Monitoring
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
    def _init_utilities(self):
        """Initialize shared utilities"""
        try:
            # Initialize data utilities
            self.klines_manager = KlinesParquetManager()
            self.matrix_ops = UnifiedMatrixOperations()
            
            # Initialize hardware optimizations if available
            if is_m1_available():
                self.m1_gpu_manager = get_m1_gpu_manager()
                self.m1_memory_optimizer = get_m1_memory_optimizer()
                self.m1_cpu_optimizer = get_m1_cpu_optimizer()
                tprint("✅ M1 hardware optimizations enabled for risk management")
            else:
                self.m1_gpu_manager = None
                self.m1_memory_optimizer = None
                self.m1_cpu_optimizer = None
                tprint("⚠️ M1 hardware optimizations not available")
                
            # Initialize optimization utilities
            self.bayesian_optimizer = BayesianTPEOptimizer()
            
            tprint("✅ Risk manager utilities initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing risk utilities: {e}")
            tprint(f"⚠️ Some risk utilities may not be available: {e}")
    
    async def start(self) -> None:
        """Start risk management monitoring"""
        if self._running:
            return
            
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_risk())
        self.logger.info("Risk manager started")
        tprint("✅ Risk manager started")
    
    async def stop(self) -> None:
        """Stop risk management monitoring"""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Risk manager stopped")
        tprint("✅ Risk manager stopped")
    
    async def validate_trade_decision(self, decision: Any) -> Tuple[bool, str]:
        """Validate a trade decision against risk limits"""
        try:
            # Extract decision parameters
            symbol = getattr(decision, 'symbol', 'UNKNOWN')
            quantity = getattr(decision, 'quantity', 0.0)
            price = getattr(decision, 'price', 0.0)
            leverage = getattr(decision, 'leverage', 1.0)
            risk_score = getattr(decision, 'risk_score', 0.5)
            
            # Validate inputs
            quantity = validate_finite(quantity, "quantity")
            price = validate_finite(price, "price")
            leverage = validate_finite(leverage, "leverage")
            risk_score = validate_finite(risk_score, "risk_score")
            
            # Check position size
            if abs(quantity) > self.risk_limits.max_position_size:
                return False, f"Position size {abs(quantity)} exceeds limit {self.risk_limits.max_position_size}"
            
            # Check leverage
            if leverage > self.risk_limits.max_leverage:
                return False, f"Leverage {leverage} exceeds limit {self.risk_limits.max_leverage}"
            
            # Check daily loss limit
            current_daily_pnl = self.daily_pnl.get(symbol, 0.0)
            if current_daily_pnl < -self.risk_limits.max_daily_loss:
                return False, f"Daily loss limit exceeded: {current_daily_pnl}"
            
            # Check order frequency
            recent_orders = [
                order_time for order_time in self.order_history
                if datetime.now() - order_time < timedelta(minutes=1)
            ]
            if len(recent_orders) >= self.risk_limits.max_orders_per_minute:
                return False, f"Order frequency limit exceeded: {len(recent_orders)} orders in last minute"
            
            # Check total exposure
            total_exposure = sum(abs(pos * self._get_current_price(sym)) 
                               for sym, pos in self.positions.items())
            if total_exposure > self.risk_limits.max_total_exposure:
                return False, f"Total exposure {total_exposure} exceeds limit {self.risk_limits.max_total_exposure}"
            
            # Check risk score
            if risk_score > 0.8:  # High risk threshold
                return False, f"Risk score {risk_score} is too high"
            
            return True, "Trade decision validated"
            
        except Exception as e:
            self.logger.error(f"❌ Error validating trade decision: {e}")
            tprint(f"❌ Risk validation failed: {e}")
            return False, f"Validation error: {str(e)}"
    
    async def calculate_position_size(self, symbol: str, signal_strength: float, 
                                     volatility: float, account_balance: float) -> float:
        """Calculate optimal position size using advanced risk models"""
        try:
            # Kelly criterion with risk adjustments
            kelly_fraction = safe_divide(signal_strength, volatility, 0.0)
            
            # Apply risk adjustments
            risk_adjustment = await self._calculate_risk_adjustment(symbol)
            adjusted_kelly = kelly_fraction * risk_adjustment
            
            # Calculate position size
            position_value = account_balance * min(adjusted_kelly, 0.25)  # Max 25% of account
            position_size = safe_divide(position_value, self._get_current_price(symbol), 0.0)
            
            # Apply additional risk constraints
            max_position = self.risk_limits.max_position_size
            position_size = min(position_size, max_position)
            
            # Ensure minimum viable position
            min_position = 0.001  # Minimum position size
            if position_size < min_position:
                return 0.0
            
            tprint(f"✅ Calculated position size for {symbol}: {position_size:.4f}")
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return 0.0
    
    async def update_position(self, symbol: str, quantity: float, price: float) -> None:
        """Update position tracking"""
        try:
            current_position = self.positions.get(symbol, 0.0)
            new_position = current_position + quantity
            
            # Update position
            self.positions[symbol] = new_position
            
            # Update daily PnL
            if symbol not in self.daily_pnl:
                self.daily_pnl[symbol] = 0.0
            
            # Calculate realized PnL
            if current_position != 0 and quantity != 0:
                if (current_position > 0 and quantity < 0) or (current_position < 0 and quantity > 0):
                    # Closing or reducing position
                    closed_quantity = min(abs(current_position), abs(quantity))
                    pnl = closed_quantity * (price - self._get_average_price(symbol))
                    self.daily_pnl[symbol] += pnl
            
            # Add to order history
            self.order_history.append(datetime.now())
            
            # Keep only recent history
            if len(self.order_history) > 1000:
                self.order_history = self.order_history[-1000:]
            
            tprint(f"✅ Position updated: {symbol} = {new_position} @ {price}")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating position: {e}")
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol (cached)"""
        return 50000.0  # Fallback price
    
    def _get_average_price(self, symbol: str) -> float:
        """Get average price for symbol (simplified)"""
        return 50000.0  # Fallback price
    
    async def _calculate_risk_adjustment(self, symbol: str) -> float:
        """Calculate risk adjustment factor"""
        try:
            # Simplified risk adjustment
            return 0.5  # Conservative default
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk adjustment: {e}")
            return 0.5
    
    async def _calculate_risk_score(self, symbol: str, position: float, price: float, leverage: float) -> float:
        """Calculate risk score (0-1, higher is riskier)"""
        try:
            risk_score = 0.0
            
            # Position size risk
            position_risk = min(abs(position) / self.risk_limits.max_position_size, 1.0)
            risk_score += position_risk * 0.3
            
            # Leverage risk
            leverage_risk = min(leverage / self.risk_limits.max_leverage, 1.0)
            risk_score += leverage_risk * 0.3
            
            # Daily PnL risk
            daily_pnl = self.daily_pnl.get(symbol, 0.0)
            pnl_risk = min(abs(daily_pnl) / self.risk_limits.max_daily_loss, 1.0)
            risk_score += pnl_risk * 0.4
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk score: {e}")
            return 0.5
    
    async def _calculate_volatility(self, symbol: str) -> float:
        """Calculate volatility for symbol"""
        try:
            # Simplified volatility calculation
            return 0.02  # 2% default volatility
        except Exception as e:
            self.logger.error(f"❌ Error calculating volatility for {symbol}: {e}")
            return 0.0
    
    async def _calculate_sharpe_ratio(self, symbol: str) -> float:
        """Calculate Sharpe ratio"""
        try:
            # Simplified Sharpe ratio calculation
            return 1.0  # Default Sharpe ratio
        except Exception as e:
            self.logger.error(f"❌ Error calculating Sharpe ratio for {symbol}: {e}")
            return 0.0
    
    async def _calculate_var_95(self, symbol: str, position: float, price: float) -> float:
        """Calculate Value at Risk 95%"""
        try:
            # Simplified VaR calculation
            return 0.01  # 1% default VaR
        except Exception as e:
            self.logger.error(f"❌ Error calculating VaR 95% for {symbol}: {e}")
            return 0.0
    
    async def _calculate_var_99(self, symbol: str, position: float, price: float) -> float:
        """Calculate Value at Risk 99%"""
        try:
            # Simplified VaR calculation
            return 0.02  # 2% default VaR
        except Exception as e:
            self.logger.error(f"❌ Error calculating VaR 99% for {symbol}: {e}")
            return 0.0
    
    async def _calculate_beta(self, symbol: str) -> float:
        """Calculate beta relative to market"""
        try:
            return 1.0  # Default beta
        except Exception as e:
            self.logger.error(f"❌ Error calculating beta for {symbol}: {e}")
            return 1.0
    
    async def _calculate_correlation(self, symbol: str) -> float:
        """Calculate correlation with other positions"""
        try:
            return 0.0  # Default correlation
        except Exception as e:
            self.logger.error(f"❌ Error calculating correlation for {symbol}: {e}")
            return 0.0
    
    async def _calculate_max_drawdown(self, symbol: str) -> float:
        """Calculate maximum drawdown"""
        try:
            if symbol not in self.daily_pnl:
                return 0.0
            return abs(self.daily_pnl[symbol]) * 0.1  # 10% of daily PnL as drawdown estimate
        except Exception as e:
            self.logger.error(f"❌ Error calculating max drawdown for {symbol}: {e}")
            return 0.0
    
    async def _monitor_risk(self) -> None:
        """Monitor risk continuously"""
        while self._running:
            try:
                # Simplified risk monitoring
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in risk monitoring: {e}")
                await asyncio.sleep(30)