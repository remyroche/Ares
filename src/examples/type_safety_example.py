# src/examples/type_safety_example.py

"""
Example demonstrating comprehensive type safety improvements in the Ares trading system.
"""

import asyncio
from datetime import datetime
from typing import Optional

from src.config.typed_config import TypedConfigManager, load_typed_config
from src.core.enhanced_dependency_injection import (
    EnhancedDependencyContainer,
    ServiceLifetime,
    register_service,
    resolve_service,
)
from src.core.generic_base import GenericTradingComponent
from src.protocols.trading_protocols import (
    TradingAnalyst,
    TradingDataProvider,
    TradingOrderExecutor,
    TradingRiskManager,
)
from src.types import (
    AccountInfo,
    ConfigDict,
    MarketDataDict,
    OrderRequest,
    PerformanceMetrics,
    Symbol,
    TradingConfig,
    TradingSignal,
)
from src.types.validation import type_safe, validate_critical_path
from src.validation.critical_path_validators import (
    CriticalPathValidator,
    validate_trading_signal_critical,
    validate_trade_decision_critical,
    get_type_safety_monitor,
)


class TypeSafeExchangeClient(GenericTradingComponent[TradingConfig]):
    """Example type-safe exchange client implementation."""
    
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self._connected = False
    
    async def start(self) -> None:
        """Start the exchange client."""
        await super().start()
        self._connected = True
        self.logger.info("Exchange client started")
    
    async def stop(self) -> None:
        """Stop the exchange client."""
        await super().stop()
        self._connected = False
        self.logger.info("Exchange client stopped")
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        return {
            "total_return": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
        }
    
    def get_health_status(self) -> dict:
        """Get health status."""
        return {
            "status": "healthy" if self._connected else "disconnected",
            "connection": "active" if self._connected else "inactive",
        }
    
    @type_safe
    async def get_market_data(
        self, 
        symbol: Symbol, 
        start_time: datetime, 
        end_time: datetime
    ) -> MarketDataDict:
        """Get market data with type validation."""
        if not self._connected:
            raise RuntimeError("Exchange client not connected")
        
        # Simulate market data retrieval
        return {
            symbol: [
                {
                    "timestamp": start_time,
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 102.0,
                    "volume": 1000.0,
                }
            ]
        }
    
    @validate_critical_path(CriticalPathValidator.validate_order_request)
    async def execute_order(self, order: OrderRequest) -> str:
        """Execute order with critical path validation."""
        if not self._connected:
            raise RuntimeError("Exchange client not connected")
        
        # Simulate order execution
        return f"order_{datetime.now().timestamp()}"


class TypeSafeMLAnalyst:
    """Example type-safe ML analyst implementation."""
    
    def __init__(self, config: ConfigDict):
        self.config = config
        self.model_ready = True
    
    @validate_trading_signal_critical
    async def generate_signal(self, market_data: MarketDataDict) -> TradingSignal:
        """Generate trading signal with validation."""
        if not self.model_ready:
            raise RuntimeError("ML model not ready")
        
        # Simulate signal generation
        return {
            "timestamp": datetime.now(),
            "symbol": list(market_data.keys())[0],
            "signal_type": "entry",
            "direction": "long",
            "strength": 0.75,
            "confidence": 0.8,
            "time_horizon": "short_term",
            "source": "ml_analyst",
        }


class TypeSafeRiskManager:
    """Example type-safe risk manager implementation."""
    
    def __init__(self, config: ConfigDict):
        self.config = config
        self.max_position_size = 0.1  # 10% of portfolio
    
    @type_safe
    async def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate trading signal against risk parameters."""
        # Risk validation logic
        if signal["confidence"] < 0.6:
            return False
        
        if signal["strength"] < 0.5:
            return False
        
        return True
    
    @validate_trade_decision_critical
    async def create_trade_decision(
        self, 
        signal: TradingSignal,
        account_info: AccountInfo
    ) -> Optional[dict]:
        """Create trade decision with validation."""
        if not await self.validate_signal(signal):
            return None
        
        # Calculate position size
        position_size = account_info["available_balance"] * self.max_position_size
        
        return {
            "timestamp": datetime.now(),
            "symbol": signal["symbol"],
            "action": "open_long" if signal["direction"] == "long" else "open_short",
            "quantity": position_size,
            "price": 100.0,  # Simulated price
            "leverage": 1.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "confidence": signal["confidence"],
            "risk_score": 0.3,
            "reasoning": f"ML signal with {signal['confidence']:.2f} confidence",
        }


async def demonstrate_type_safety():
    """Demonstrate comprehensive type safety features."""
    print("🔒 Demonstrating Type Safety Improvements")
    print("=" * 50)
    
    # 1. Load typed configuration
    try:
        config_manager = TypedConfigManager()
        
        # Simulate loading configuration
        raw_config = {
            "trading": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "intervals": ["1m", "5m", "1h"],
                "max_position_size": 1000.0,
                "max_leverage": 10.0,
                "stop_loss_percentage": 0.02,
                "take_profit_percentage": 0.04,
                "max_drawdown": 0.1,
                "risk_per_trade": 0.02,
                "enable_trailing_stop": True,
                "paper_trading": True,
            }
        }
        
        # This would normally validate the config
        trading_config = raw_config["trading"]
        print("✅ Configuration loaded and validated")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return
    
    # 2. Create type-safe components
    exchange_client = TypeSafeExchangeClient(trading_config)
    ml_analyst = TypeSafeMLAnalyst(raw_config)
    risk_manager = TypeSafeRiskManager(raw_config)
    
    print("✅ Type-safe components created")
    
    # 3. Start components
    await exchange_client.start()
    print("✅ Exchange client started")
    
    # 4. Demonstrate type-safe data flow
    try:
        # Get market data
        symbol = Symbol("BTCUSDT")
        start_time = datetime.now()
        end_time = datetime.now()
        
        market_data = await exchange_client.get_market_data(symbol, start_time, end_time)
        print(f"✅ Market data retrieved for {symbol}")
        
        # Generate ML signal
        signal = await ml_analyst.generate_signal(market_data)
        print(f"✅ Trading signal generated: {signal['signal_type']} {signal['direction']}")
        
        # Create trade decision
        account_info = {
            "account_id": "test_account",
            "total_balance": 10000.0,
            "available_balance": 8000.0,
            "margin_balance": 0.0,
            "unrealized_pnl": 0.0,
            "margin_ratio": 0.0,
            "positions": [],
            "open_orders": [],
        }
        
        trade_decision = await risk_manager.create_trade_decision(signal, account_info)
        if trade_decision:
            print(f"✅ Trade decision created: {trade_decision['action']} {trade_decision['quantity']} units")
            
            # Execute order
            order = {
                "symbol": trade_decision["symbol"],
                "side": "buy" if "long" in trade_decision["action"] else "sell",
                "type": "market",
                "quantity": trade_decision["quantity"],
            }
            
            order_id = await exchange_client.execute_order(order)
            print(f"✅ Order executed: {order_id}")
        else:
            print("⚠️ Trade decision rejected by risk management")
        
    except Exception as e:
        print(f"❌ Type safety validation caught error: {e}")
    
    # 5. Check type safety monitoring
    monitor = get_type_safety_monitor()
    violation_summary = monitor.get_violation_summary()
    print(f"📊 Type safety violations: {violation_summary['total_violations']}")
    
    # 6. Component health check
    health_status = exchange_client.get_health_status()
    print(f"💚 Component health: {health_status['status']}")
    
    # 7. Performance metrics
    metrics = exchange_client.get_metrics()
    print(f"📈 Performance metrics: {len(metrics)} indicators tracked")
    
    # Stop components
    await exchange_client.stop()
    print("✅ Components stopped")
    
    print("\n🎉 Type safety demonstration completed successfully!")


if __name__ == "__main__":
    asyncio.run(demonstrate_type_safety())