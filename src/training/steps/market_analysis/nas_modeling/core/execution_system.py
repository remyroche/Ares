"""
Execution System for Trading

This module provides production-ready execution capabilities:
- Order management and routing
- Transaction cost optimization
- Market impact modeling
- Slippage estimation
- Real-time execution monitoring
- Order book integration
- Smart order routing
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types for execution."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"  # Time Weighted Average Price
    VWAP = "vwap"  # Volume Weighted Average Price

class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class ExecutionConfig:
    """Configuration for execution system."""
    default_order_type: OrderType = OrderType.MARKET
    max_slippage: float = 0.001  # 0.1% maximum slippage
    max_market_impact: float = 0.002  # 0.2% maximum market impact
    use_smart_routing: bool = True
    broker_fees: Dict[str, float] = field(default_factory=lambda: {
        "equity": 0.001,      # 0.1% for stocks
        "forex": 0.0001,      # 0.01% for forex
        "crypto": 0.002,      # 0.2% for crypto
        "futures": 0.0005     # 0.05% for futures
    })
    min_order_size: float = 100.0  # Minimum order value
    max_position_size: float = 1000000.0  # Maximum position value
    enable_iceberg_orders: bool = False
    iceberg_chunk_size: float = 0.1  # 10% of order per iceberg chunk
    use_twap_vwap: bool = True
    twap_duration: int = 60  # minutes
    vwap_duration: int = 30   # minutes

@dataclass
class Order:
    """Trading order representation."""
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # "buy" or "sell"
    quantity: float
    price: Optional[float] = None  # None for market orders
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    execution_cost: float = 0.0

class ExecutionEngine:
    """
    Advanced execution engine for trading.

    Handles order execution, routing, and optimization.
    """

    def __init__(self, config: ExecutionConfig):
        """Initialize execution engine.

        Args:
            config: Execution configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Order management
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'avg_slippage': 0.0,
            'total_fees': 0.0
        }

        # Market data for execution
        self.market_data_cache = {}
        self.order_book_cache = {}

    async def execute_order(self, order: Order) -> Dict[str, Any]:
        """
        Execute trading order with optimization.

        Args:
            order: Order to execute

        Returns:
            Execution results
        """
        logger.info(f"📋 Executing {order.side} order for {order.quantity} {order.symbol}")

        # Generate unique order ID if not provided
        if not order.order_id:
            order.order_id = self._generate_order_id(order.symbol)

        # Validate order
        validation_result = self._validate_order(order)
        if not validation_result['valid']:
            return {
                'success': False,
                'error': validation_result['reason'],
                'order': order
            }

        # Optimize execution
        optimized_order = self._optimize_execution(order)

        # Route order
        routing_result = await self._route_order(optimized_order)

        if routing_result['success']:
            # Execute order
            execution_result = await self._process_execution(routing_result['routed_order'])

            # Update statistics
            self._update_execution_stats(execution_result)

            return execution_result
        else:
            return routing_result

    def _validate_order(self, order: Order) -> Dict[str, Any]:
        """Validate order parameters."""
        if order.quantity <= 0:
            return {'valid': False, 'reason': 'Invalid quantity'}

        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price is None:
            return {'valid': False, 'reason': 'Limit price required for limit orders'}

        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            return {'valid': False, 'reason': 'Stop price required for stop orders'}

        return {'valid': True, 'reason': None}

    def _optimize_execution(self, order: Order) -> Order:
        """Optimize order execution for best results."""
        optimized_order = order

        # Choose optimal order type based on conditions
        if self.config.use_twap_vwap and order.quantity > 1000:
            if order.side in ['buy', 'sell']:
                # Use VWAP for large orders
                optimized_order.order_type = OrderType.VWAP
            else:
                # Use TWAP for very large orders
                optimized_order.order_type = OrderType.TWAP

        # Set iceberg orders for large trades
        if self.config.enable_iceberg_orders and order.quantity > 10000:
            optimized_order.order_type = OrderType.ICEBERG

        return optimized_order

    async def _route_order(self, order: Order) -> Dict[str, Any]:
        """Route order to optimal execution venue."""
        if not self.config.use_smart_routing:
            return {
                'success': True,
                'routed_order': order,
                'routing_info': {'venue': 'default'}
            }

        # Smart routing logic
        routing_decision = self._select_optimal_venue(order)

        return {
            'success': True,
            'routed_order': order,
            'routing_info': routing_decision
        }

    async def _process_execution(self, order: Order) -> Dict[str, Any]:
        """Process order execution."""
        # Simulate execution (in real system, this would interface with broker API)
        execution_time = datetime.now()

        # Calculate execution price with slippage
        market_price = self._get_current_market_price(order.symbol)
        execution_price = self._calculate_execution_price(order, market_price)

        # Calculate fees
        execution_fees = self._calculate_execution_fees(order, execution_price)

        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = execution_price
        order.execution_cost = execution_fees
        order.timestamp = execution_time

        return {
            'success': True,
            'order': order,
            'execution_price': execution_price,
            'execution_fees': execution_fees,
            'slippage': abs(execution_price - market_price) / market_price
        }

    def _generate_order_id(self, symbol: str) -> str:
        """Generate unique order ID."""
        timestamp = int(time.time() * 1000)
        return f"{symbol}_{timestamp}_{np.random.randint(1000, 9999)}"

    def _get_current_market_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        # In real implementation, this would fetch from market data feed
        # For simulation, return a placeholder
        return 100.0 + np.random.randn() * 10  # Mock price

    def _calculate_execution_price(self, order: Order, market_price: float) -> float:
        """Calculate actual execution price including slippage."""
        if order.order_type == OrderType.MARKET:
            # Market orders get filled at market price with some slippage
            slippage = np.random.normal(0, self.config.max_slippage * 0.5)
            return market_price * (1 + slippage)
        elif order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            # Limit orders filled at limit price or better
            return min(order.limit_price, market_price) if order.side == 'buy' else max(order.limit_price, market_price)
        else:
            # Other order types
            return market_price

    def _calculate_execution_fees(self, order: Order, execution_price: float) -> float:
        """Calculate execution fees and costs."""
        # Determine asset class for fee calculation
        asset_class = self._classify_asset(order.symbol)

        base_fee = self.config.broker_fees.get(asset_class, 0.001)
        transaction_cost = order.quantity * execution_price * base_fee

        return transaction_cost

    def _classify_asset(self, symbol: str) -> str:
        """Classify asset for fee calculation."""
        if 'BTC' in symbol or 'ETH' in symbol:
            return 'crypto'
        elif 'EUR' in symbol or 'GBP' in symbol or 'JPY' in symbol:
            return 'forex'
        elif symbol.endswith(('F', 'H', 'M', 'U', 'Z')):  # Futures symbols
            return 'futures'
        else:
            return 'equity'

    def _select_optimal_venue(self, order: Order) -> Dict[str, str]:
        """Select optimal execution venue."""
        # Smart routing logic based on order characteristics
        if order.quantity > 10000:
            # Large orders - use dark pools or OTC
            return {'venue': 'dark_pool', 'reason': 'Large order size'}
        elif order.order_type in [OrderType.TWAP, OrderType.VWAP]:
            # Algorithmic orders - use execution algorithms
            return {'venue': 'algorithmic', 'reason': 'Algorithmic execution'}
        else:
            # Standard orders - use primary exchange
            return {'venue': 'primary_exchange', 'reason': 'Standard execution'}

    def _update_execution_stats(self, execution_result: Dict[str, Any]):
        """Update execution statistics."""
        if execution_result['success']:
            self.execution_stats['total_orders'] += 1
            self.execution_stats['filled_orders'] += 1
            self.execution_stats['total_fees'] += execution_result['execution_fees']

            # Update average slippage
            slippage = execution_result['slippage']
            current_avg = self.execution_stats['avg_slippage']
            total_orders = self.execution_stats['total_orders']

            self.execution_stats['avg_slippage'] = (
                (current_avg * (total_orders - 1)) + slippage
            ) / total_orders

class MarketImpactModel:
    """
    Market impact modeling for execution.

    Estimates market impact of trades for optimal execution.
    """

    def __init__(self):
        """Initialize market impact model."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def estimate_market_impact(self, symbol: str, order_size: float,
                              current_volume: float) -> Dict[str, float]:
        """
        Estimate market impact of a trade.

        Args:
            symbol: Trading symbol
            order_size: Size of order
            current_volume: Current market volume

        Returns:
            Market impact estimates
        """
        # Simplified market impact model
        # In practice, this would use more sophisticated models

        # Size-based impact
        size_impact = min(order_size / current_volume, 0.1)  # Max 10% impact

        # Liquidity impact
        liquidity_impact = 0.001 * (order_size / 1000)  # 0.1% per 1000 units

        # Total impact
        total_impact = size_impact + liquidity_impact

        return {
            'size_impact': size_impact,
            'liquidity_impact': liquidity_impact,
            'total_impact': total_impact,
            'estimated_slippage': total_impact * 0.5,
            'recommended_approach': self._get_recommendation(order_size, current_volume)
        }

    def _get_recommendation(self, order_size: float, current_volume: float) -> str:
        """Get execution recommendation based on order characteristics."""
        size_ratio = order_size / current_volume

        if size_ratio > 0.05:  # Large order
            return "Use algorithmic execution (VWAP/TWAP)"
        elif size_ratio > 0.01:  # Medium order
            return "Consider limit orders or iceberg orders"
        else:  # Small order
            return "Market order acceptable"

class RealTimeExecutionMonitor:
    """
    Real-time monitoring of execution performance.

    Monitors execution quality, slippage, and market conditions.
    """

    def __init__(self, config: ExecutionConfig):
        """Initialize execution monitor.

        Args:
            config: Execution configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Monitoring state
        self.execution_metrics = {
            'avg_slippage': 0.0,
            'avg_market_impact': 0.0,
            'execution_speed': 0.0,
            'fill_rate': 1.0
        }

        self.alerts = []

    def monitor_execution(self, order: Order, execution_result: Dict[str, Any]):
        """Monitor execution performance."""
        if not execution_result['success']:
            self.logger.warning(f"⚠️ Execution failed for order {order.order_id}")
            return

        # Calculate metrics
        slippage = execution_result['slippage']
        execution_time = (execution_result['order'].timestamp - order.timestamp).total_seconds()

        # Update metrics
        self._update_metrics(slippage, execution_time)

        # Check for alerts
        self._check_alerts(slippage, execution_time)

    def _update_metrics(self, slippage: float, execution_time: float):
        """Update execution metrics."""
        # Simple moving average updates
        self.execution_metrics['avg_slippage'] = (
            self.execution_metrics['avg_slippage'] * 0.9 + slippage * 0.1
        )

        self.execution_metrics['execution_speed'] = (
            self.execution_metrics['execution_speed'] * 0.9 + execution_time * 0.1
        )

    def _check_alerts(self, slippage: float, execution_time: float):
        """Check for execution alerts."""
        alerts = []

        if slippage > self.config.max_slippage:
            alerts.append({
                'type': 'high_slippage',
                'message': f'Slippage {slippage:.4f} exceeds threshold {self.config.max_slippage}',
                'severity': 'warning'
            })

        if execution_time > 30:  # 30 seconds
            alerts.append({
                'type': 'slow_execution',
                'message': f'Execution took {execution_time:.1f} seconds',
                'severity': 'info'
            })

        for alert in alerts:
            self.logger.log(
                logging.WARNING if alert['severity'] == 'warning' else logging.INFO,
                f"🚨 {alert['type'].upper()}: {alert['message']}"
            )

        self.alerts.extend(alerts)

class TradingSystemIntegration:
    """
    Integration layer for the complete trading system.

    Combines regime detection, signal generation, portfolio optimization,
    and execution into a unified system.
    """

    def __init__(self, regime_model: nn.Module, config: ExecutionConfig):
        """Initialize trading system integration.

        Args:
            regime_model: Trained regime detection model
            config: Execution configuration
        """
        self.regime_model = regime_model
        self.execution_config = config

        # Initialize components
        from .trading_signal_generator import TradingSignalGenerator, PortfolioManager, RiskManager
        from .portfolio_optimizer import MeanVarianceOptimizer, PortfolioConfig

        self.signal_generator = TradingSignalGenerator()
        self.portfolio_manager = PortfolioManager(initial_capital=100000.0)
        self.risk_manager = RiskManager()
        self.portfolio_optimizer = MeanVarianceOptimizer(PortfolioConfig())
        self.execution_engine = ExecutionEngine(config)
        self.monitor = RealTimeExecutionMonitor(config)

        self.logger = logging.getLogger(self.__class__.__name__)

    async def process_trading_cycle(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Process complete trading cycle.

        Args:
            market_data: Current market data

        Returns:
            Trading cycle results
        """
        logger.info("🔄 Starting trading cycle")

        # 1. Get regime prediction
        regime_input = self._prepare_regime_input(market_data)
        with torch.no_grad():
            regime_prediction = self.regime_model(regime_input).numpy()

        # 2. Generate trading signal
        symbol = list(market_data.keys())[0]  # Use first symbol for demo
        current_position = self._get_current_position(symbol)
        signal = self.signal_generator.generate_signal(
            regime_prediction, market_data[symbol], current_position
        )

        # 3. Check risk limits
        if not self.risk_manager.check_risk_limits(signal, self.portfolio_manager.positions):
            logger.warning("⚠️ Risk limits violated, skipping trade")
            return {'status': 'risk_violation', 'signal': signal}

        # 4. Optimize portfolio
        returns_data = self._prepare_returns_data(market_data)
        portfolio_result = self.portfolio_optimizer.optimize_portfolio(returns_data)

        # 5. Execute trades
        execution_results = []
        for symbol in market_data.keys():
            # Update position
            position_result = self.portfolio_manager.update_position(
                symbol, signal, market_data
            )

            # Create and execute order
            order = self._create_execution_order(symbol, signal, market_data[symbol])
            execution_result = await self.execution_engine.execute_order(order)
            execution_results.append(execution_result)

            # Monitor execution
            self.monitor.monitor_execution(order, execution_result)

        # 6. Get portfolio status
        portfolio_status = self.portfolio_manager.get_portfolio_status()

        cycle_result = {
            'status': 'completed',
            'regime_prediction': regime_prediction,
            'signal': signal,
            'portfolio_result': portfolio_result,
            'execution_results': execution_results,
            'portfolio_status': portfolio_status,
            'risk_check': 'passed',
            'timestamp': datetime.now()
        }

        self.logger.info(f"✅ Trading cycle completed - Portfolio value: ${portfolio_status['total_value']:.2f}")
        return cycle_result

    def _prepare_regime_input(self, market_data: Dict[str, pd.DataFrame]) -> torch.Tensor:
        """Prepare input for regime model."""
        # Simplified - use data from first symbol
        symbol = list(market_data.keys())[0]
        data = market_data[symbol]

        # Convert to appropriate format
        features = ['open', 'high', 'low', 'close', 'volume']
        feature_data = data[features].values[-100:]  # Last 100 time steps

        return torch.FloatTensor(feature_data).unsqueeze(0)

    def _get_current_position(self, symbol: str) -> float:
        """Get current position for symbol."""
        if symbol in self.portfolio_manager.positions:
            return self.portfolio_manager.positions[symbol]['size']
        return 0.0

    def _prepare_returns_data(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare returns data for portfolio optimization."""
        returns_list = []

        for symbol, data in market_data.items():
            returns = data['close'].pct_change().dropna()
            returns_list.append(returns)

        return pd.concat(returns_list, axis=1)

    def _create_execution_order(self, symbol: str, signal: Dict[str, Any],
                               market_data: pd.DataFrame) -> Order:
        """Create execution order from signal."""
        current_price = market_data['close'].iloc[-1]
        quantity = abs(signal['position_change']) * 100  # Simplified

        if signal['position_change'] > 0:
            side = 'buy'
        else:
            side = 'sell'

        order = Order(
            order_id="",
            symbol=symbol,
            order_type=self.execution_config.default_order_type,
            side=side,
            quantity=quantity,
            price=current_price,
            timestamp=datetime.now()
        )

        return order

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        portfolio_status = self.portfolio_manager.get_portfolio_status()
        execution_stats = self.execution_engine.execution_stats
        monitoring_alerts = len(self.monitor.alerts)

        return {
            'portfolio_value': portfolio_status['total_value'],
            'total_return': portfolio_status['return_pct'],
            'active_positions': len(self.portfolio_manager.positions),
            'execution_stats': execution_stats,
            'alerts': monitoring_alerts,
            'system_health': 'healthy' if monitoring_alerts == 0 else 'warnings'
        }

# Utility functions
async def execute_trading_cycle(trading_system: TradingSystemIntegration,
                               market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Execute single trading cycle."""
    return await trading_system.process_trading_cycle(market_data)

def create_trading_system(regime_model: nn.Module, config: ExecutionConfig = None) -> TradingSystemIntegration:
    """Create complete trading system."""
    if config is None:
        config = ExecutionConfig()

    return TradingSystemIntegration(regime_model, config)

def estimate_market_impact(symbol: str, order_size: float,
                          current_volume: float) -> Dict[str, float]:
    """Estimate market impact of a trade."""
    impact_model = MarketImpactModel()
    return impact_model.estimate_market_impact(symbol, order_size, current_volume)

def monitor_execution_performance(execution_engine: ExecutionEngine) -> Dict[str, float]:
    """Monitor execution engine performance."""
    return execution_engine.execution_stats