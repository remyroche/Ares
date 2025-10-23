"""
Risk Management System

Handles risk management for live trading operations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

from .config import TradingConfig
from .order_manager import Order, OrderStatus
from ..src.interfaces.base_interfaces import TradeDecision, MarketData


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
    max_position_size: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None


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


class RiskManager:
    """Manages risk for live trading operations"""
    
    def __init__(self, config: TradingConfig, exchange_client: Any):
        self.config = config
        self.exchange_client = exchange_client
        self.logger = logging.getLogger(__name__)
        
        # Risk limits
        self.risk_limits = RiskLimits(
            max_position_size=config.max_position_size,
            max_daily_loss=config.max_daily_loss,
            max_leverage=config.max_leverage,
            max_drawdown_percent=10.0,  # 10% max drawdown
            max_volatility=0.05,  # 5% max volatility
            min_sharpe_ratio=1.0,
            max_correlation=0.8,
            max_orders_per_minute=60,
            max_total_exposure=config.max_position_size * 5  # 5x max position size
        )
        
        # Risk tracking
        self.risk_metrics: Dict[str, RiskMetrics] = {}
        self.daily_pnl: Dict[str, float] = {}
        self.order_history: List[datetime] = []
        self.positions: Dict[str, float] = {}
        
        # Risk events
        self.risk_violations: List[Dict[str, Any]] = []
        self.risk_alerts: List[Dict[str, Any]] = []
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start risk management monitoring"""
        if self._running:
            return
            
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_risk())
        self.logger.info("Risk manager started")
    
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
    
    async def validate_trade_decision(self, decision: TradeDecision) -> Tuple[bool, str]:
        """Validate a trade decision against risk limits"""
        try:
            # Check position size
            if abs(decision.quantity) > self.risk_limits.max_position_size:
                return False, f"Position size {decision.quantity} exceeds limit {self.risk_limits.max_position_size}"
            
            # Check leverage
            if decision.leverage > self.risk_limits.max_leverage:
                return False, f"Leverage {decision.leverage} exceeds limit {self.risk_limits.max_leverage}"
            
            # Check daily loss limit
            current_daily_pnl = self.daily_pnl.get(decision.symbol, 0.0)
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
            total_exposure = sum(abs(pos * self._get_current_price(symbol)) 
                               for symbol, pos in self.positions.items())
            if total_exposure > self.risk_limits.max_total_exposure:
                return False, f"Total exposure {total_exposure} exceeds limit {self.risk_limits.max_total_exposure}"
            
            # Check risk score
            if decision.risk_score > 0.8:  # High risk threshold
                return False, f"Risk score {decision.risk_score} is too high"
            
            # Check volatility if available
            if hasattr(decision, 'volatility') and decision.volatility > self.risk_limits.max_volatility:
                return False, f"Volatility {decision.volatility} exceeds limit {self.risk_limits.max_volatility}"
            
            return True, "Trade decision validated"
            
        except Exception as e:
            self.logger.error(f"❌ Error validating trade decision: {e}")
            self.logger.warning("⚠️ Trade validation failed - rejecting trade to ensure safety")
            return False, f"Validation error: {str(e)}"
    
    async def update_position(self, symbol: str, quantity: float, price: float) -> None:
        """Update position tracking"""
        current_position = self.positions.get(symbol, 0.0)
        new_position = current_position + quantity
        
        # Update position
        self.positions[symbol] = new_position
        
        # Update daily PnL
        if symbol not in self.daily_pnl:
            self.daily_pnl[symbol] = 0.0
        
        # Calculate realized PnL (simplified)
        if current_position != 0 and quantity != 0:
            if (current_position > 0 and quantity < 0) or (current_position < 0 and quantity > 0):
                # Closing or reducing position
                closed_quantity = min(abs(current_position), abs(quantity))
                pnl = closed_quantity * (price - self._get_average_price(symbol))
                self.daily_pnl[symbol] += pnl
        
        # Log position update
        self.logger.info(f"Position updated: {symbol} = {new_position} @ {price}")
    
    async def calculate_risk_metrics(self, symbol: str) -> RiskMetrics:
        """Calculate comprehensive risk metrics for a symbol"""
        try:
            # Get current position
            current_position = self.positions.get(symbol, 0.0)
            
            # Get current market data
            ticker = await self.exchange_client.get_ticker(symbol)
            current_price = float(ticker.get("last", 0)) if ticker else 0.0
            
            # Calculate position value
            position_value = abs(current_position) * current_price
            
            # Get account info for margin calculations
            account_info = await self.exchange_client.get_account_info()
            total_balance = float(account_info.get("totalBalance", 0))
            available_balance = float(account_info.get("availableBalance", 0))
            
            # Calculate leverage (simplified)
            leverage = position_value / total_balance if total_balance > 0 else 0.0
            
            # Calculate daily PnL
            daily_pnl = self.daily_pnl.get(symbol, 0.0)
            
            # Calculate risk score (simplified)
            risk_score = self._calculate_risk_score(symbol, current_position, current_price, leverage)
            
            # Calculate volatility (simplified)
            volatility = await self._calculate_volatility(symbol)
            
            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = await self._calculate_sharpe_ratio(symbol)
            
            # Create risk metrics
            metrics = RiskMetrics(
                timestamp=datetime.now(),
                symbol=symbol,
                current_position=current_position,
                position_value=position_value,
                unrealized_pnl=0.0,  # Would need more sophisticated calculation
                realized_pnl=daily_pnl,
                daily_pnl=daily_pnl,
                max_drawdown=0.0,  # Would need historical data
                leverage=leverage,
                margin_used=position_value,
                margin_available=available_balance,
                risk_score=risk_score,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_position_size=self.risk_limits.max_position_size
            )
            
            # Store metrics
            self.risk_metrics[symbol] = metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk metrics for {symbol}: {e}")
            self.logger.warning(f"⚠️ Using default risk metrics for {symbol} - risk assessment may be inaccurate")
            # Return default metrics
            return RiskMetrics(
                timestamp=datetime.now(),
                symbol=symbol,
                current_position=0.0,
                position_value=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                daily_pnl=0.0,
                max_drawdown=0.0,
                leverage=0.0,
                margin_used=0.0,
                margin_available=0.0,
                risk_score=0.5,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_position_size=self.risk_limits.max_position_size
            )
    
    async def check_risk_limits(self, symbol: str) -> List[str]:
        """Check if any risk limits are violated"""
        violations = []
        
        try:
            metrics = await self.calculate_risk_metrics(symbol)
            
            # Check position size
            if abs(metrics.current_position) > self.risk_limits.max_position_size:
                violations.append(f"Position size {metrics.current_position} exceeds limit")
            
            # Check daily loss
            if metrics.daily_pnl < -self.risk_limits.max_daily_loss:
                violations.append(f"Daily loss {metrics.daily_pnl} exceeds limit")
            
            # Check leverage
            if metrics.leverage > self.risk_limits.max_leverage:
                violations.append(f"Leverage {metrics.leverage} exceeds limit")
            
            # Check volatility
            if metrics.volatility > self.risk_limits.max_volatility:
                violations.append(f"Volatility {metrics.volatility} exceeds limit")
            
            # Check Sharpe ratio
            if metrics.sharpe_ratio < self.risk_limits.min_sharpe_ratio:
                violations.append(f"Sharpe ratio {metrics.sharpe_ratio} below minimum")
            
            # Log violations
            if violations:
                violation_data = {
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "violations": violations,
                    "metrics": metrics
                }
                self.risk_violations.append(violation_data)
                self.logger.warning(f"Risk violations for {symbol}: {violations}")
            
        except Exception as e:
            self.logger.error(f"❌ Error checking risk limits for {symbol}: {e}")
            self.logger.warning("⚠️ Risk limits check failed - assuming no violations to allow trading")
            violations.append(f"Error checking risk limits: {str(e)}")
        
        return violations
    
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        summary = {
            "timestamp": datetime.now(),
            "total_positions": len(self.positions),
            "total_exposure": 0.0,
            "total_daily_pnl": 0.0,
            "risk_violations_count": len(self.risk_violations),
            "risk_alerts_count": len(self.risk_alerts),
            "positions": {},
            "risk_metrics": {},
            "limits": {
                "max_position_size": self.risk_limits.max_position_size,
                "max_daily_loss": self.risk_limits.max_daily_loss,
                "max_leverage": self.risk_limits.max_leverage,
                "max_total_exposure": self.risk_limits.max_total_exposure
            }
        }
        
        # Calculate totals
        for symbol, position in self.positions.items():
            ticker = await self.exchange_client.get_ticker(symbol)
            current_price = float(ticker.get("last", 0)) if ticker else 0.0
            position_value = abs(position) * current_price
            daily_pnl = self.daily_pnl.get(symbol, 0.0)
            
            summary["total_exposure"] += position_value
            summary["total_daily_pnl"] += daily_pnl
            
            summary["positions"][symbol] = {
                "position": position,
                "value": position_value,
                "daily_pnl": daily_pnl
            }
            
            # Get risk metrics
            if symbol in self.risk_metrics:
                summary["risk_metrics"][symbol] = {
                    "risk_score": self.risk_metrics[symbol].risk_score,
                    "leverage": self.risk_metrics[symbol].leverage,
                    "volatility": self.risk_metrics[symbol].volatility,
                    "sharpe_ratio": self.risk_metrics[symbol].sharpe_ratio
                }
        
        return summary
    
    async def _monitor_risk(self) -> None:
        """Monitor risk continuously"""
        while self._running:
            try:
                # Check risk limits for all symbols
                for symbol in self.config.symbols:
                    violations = await self.check_risk_limits(symbol)
                    
                    if violations:
                        # Generate risk alert
                        alert = {
                            "timestamp": datetime.now(),
                            "symbol": symbol,
                            "type": "risk_violation",
                            "message": f"Risk violations detected: {', '.join(violations)}",
                            "severity": "high"
                        }
                        self.risk_alerts.append(alert)
                        self.logger.warning(f"Risk alert: {alert['message']}")
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in risk monitoring: {e}")
                self.logger.warning("⚠️ Risk monitoring loop failed - continuing after retry delay")
                await asyncio.sleep(30)
    
    def _calculate_risk_score(self, symbol: str, position: float, price: float, leverage: float) -> float:
        """Calculate risk score (0-1, higher is riskier)"""
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
        risk_score += pnl_risk * 0.2
        
        # Volatility risk (simplified)
        volatility = self.risk_metrics.get(symbol, RiskMetrics(
            timestamp=datetime.now(), symbol=symbol, current_position=0.0,
            position_value=0.0, unrealized_pnl=0.0, realized_pnl=0.0,
            daily_pnl=0.0, max_drawdown=0.0, leverage=0.0, margin_used=0.0,
            margin_available=0.0, risk_score=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_position_size=0.0
        )).volatility
        vol_risk = min(volatility / self.risk_limits.max_volatility, 1.0)
        risk_score += vol_risk * 0.2
        
        return min(risk_score, 1.0)
    
    async def _calculate_volatility(self, symbol: str) -> float:
        """Calculate volatility for symbol (simplified)"""
        try:
            # Get recent klines for volatility calculation
            klines = await self.exchange_client.get_klines(symbol, "1h", limit=24)
            
            if len(klines) < 2:
                return 0.0
            
            # Calculate returns
            returns = []
            for i in range(1, len(klines)):
                prev_close = klines[i-1].close
                curr_close = klines[i].close
                if prev_close > 0:
                    returns.append((curr_close - prev_close) / prev_close)
            
            if not returns:
                return 0.0
            
            # Calculate standard deviation
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = variance ** 0.5
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating volatility for {symbol}: {e}")
            self.logger.warning(f"⚠️ Using zero volatility for {symbol} - risk calculations may be inaccurate")
            return 0.0
    
    async def _calculate_sharpe_ratio(self, symbol: str) -> float:
        """Calculate Sharpe ratio (simplified)"""
        try:
            # Get recent klines
            klines = await self.exchange_client.get_klines(symbol, "1d", limit=30)
            
            if len(klines) < 2:
                return 0.0
            
            # Calculate returns
            returns = []
            for i in range(1, len(klines)):
                prev_close = klines[i-1].close
                curr_close = klines[i].close
                if prev_close > 0:
                    returns.append((curr_close - prev_close) / prev_close)
            
            if not returns:
                return 0.0
            
            # Calculate Sharpe ratio (simplified)
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5
            
            # Risk-free rate assumed to be 0
            sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0.0
            
            return sharpe_ratio
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating Sharpe ratio for {symbol}: {e}")
            self.logger.warning(f"⚠️ Using zero Sharpe ratio for {symbol} - risk calculations may be inaccurate")
            return 0.0
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol (cached)"""
        # This would typically use cached data from data streamer
        return 50000.0  # Fallback price
    
    def _get_average_price(self, symbol: str) -> float:
        """Get average price for symbol (simplified)"""
        # This would typically track average entry price
        return 50000.0  # Fallback price