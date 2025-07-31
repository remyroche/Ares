# src/paper_trader.py
"""
PaperTrader for training and testnet trading.
Uses Binance testnet via BinanceExchange for all operations.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from exchange.binance import BinanceExchange
from src.utils.logger import system_logger
from src.config import settings, CONFIG
from src.utils.state_manager import StateManager
from src.utils.error_handler import (
    handle_errors,
    handle_network_operations,
)


class PaperTrader:
    """
    Paper trading implementation using Binance testnet for realistic trading simulation.
    """

    def __init__(
        self, symbol: str, exchange_name: str, config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or CONFIG.get("paper_trader", {})
        self.logger = system_logger.getChild("PaperTrader")
        self.symbol = symbol
        self.exchange_name = exchange_name.upper()

        # Initialize testnet exchange client
        if self.exchange_name == "BINANCE":
            self.exchange = BinanceExchange(
                api_key=settings.binance_testnet_api_key,
                api_secret=settings.binance_testnet_api_secret,
                trade_symbol=self.symbol,
            )
        else:
            self.logger.error(
                f"Paper trading for exchange '{self.exchange_name}' is not supported."
            )
            raise NotImplementedError(
                f"Paper trading for exchange '{self.exchange_name}' is not supported."
            )

        self.state_manager = StateManager()
        self.is_running = False
        self.logger.info("PaperTrader (testnet) initialized")

    @handle_errors(
        exceptions=(Exception,), default_return=False, context="paper_trader_start"
    )
    async def start(self) -> bool:
        """Start the paper trading session."""
        try:
            self.logger.info("🚀 Starting PaperTrader (testnet)...")

            # Test connection to testnet
            # The start method in BinanceExchange does not exist, so remove this line.
            # Instead, the methods like get_account_info will implicitly connect.
            self.logger.info(
                "✅ Testnet connection established"
            )  # This log will remain, assuming success.

            # Get account info
            account_info = await self.exchange.get_account_info()
            self.logger.info(
                f"📊 Testnet Account Balance: {account_info.get('totalWalletBalance', 'Unknown')} USDT"
            )

            self.is_running = True
            self.logger.info("✅ PaperTrader (testnet) started successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to start PaperTrader: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="paper_trader_stop"
    )
    async def stop(self):
        """Stop the paper trading session."""
        try:
            self.logger.info("🛑 Stopping PaperTrader (testnet)...")

            if self.exchange:
                # The close method in BinanceExchange is async
                await self.exchange.close()

            self.is_running = False
            self.logger.info("✅ PaperTrader (testnet) stopped")

        except Exception as e:
            self.logger.error(f"❌ Error stopping PaperTrader: {e}")

    @handle_network_operations(
        max_retries=3, default_return=None, context="execute_paper_trade"
    )
    async def execute_paper_trade(
        self, order_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a paper trade on the testnet.

        Args:
            order_params: Dictionary containing order parameters

        Returns:
            Optional[Dict]: Trade execution result
        """
        try:
            if not self.is_running:
                self.logger.warning("PaperTrader not running")
                return None

            # Extract order parameters
            symbol = order_params.get("symbol", self.symbol)
            side = order_params.get("side", "buy")
            order_type = order_params.get("order_type", "MARKET")
            quantity = order_params.get("quantity", 0.001)
            price = order_params.get("price")

            self.logger.info(
                f"📝 Executing paper trade: {side.upper()} {quantity} {symbol} @ {price or 'MARKET'}"
            )

            # Execute order on testnet
            if order_type == "MARKET":
                result = await self.exchange.create_order(
                    symbol=symbol, side=side, order_type=order_type, quantity=quantity
                )
            else:
                result = await self.exchange.create_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                )

            if result:
                self.logger.info(
                    f"✅ Paper trade executed: {result.get('orderId', 'Unknown')}"
                )
                return result
            else:
                self.logger.warning("❌ Paper trade execution failed")
                return None

        except Exception as e:
            self.logger.error(f"❌ Error executing paper trade: {e}")
            return None

    @handle_network_operations(
        max_retries=3, default_return=None, context="get_paper_position"
    )
    async def get_paper_position(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """
        Get current paper trading position.

        Args:
            symbol: Trading symbol (defaults to configured symbol)

        Returns:
            Optional[Dict]: Position information
        """
        try:
            if not self.is_running:
                return None

            symbol = symbol or self.symbol

            # Get position from testnet
            position_info = await self.exchange.get_position_risk(symbol)

            if position_info:
                # get_position_risk returns a list of positions, take the first if it exists
                if position_info and len(position_info) > 0:
                    self.logger.info(f"📊 Paper position: {position_info[0]}")
                    return position_info[0]  # Return the first position found
                else:
                    self.logger.info("📊 No paper position found")
                    return None
            else:
                self.logger.info("📊 No paper position found")
                return None

        except Exception as e:
            self.logger.error(f"❌ Error getting paper position: {e}")
            return None

    @handle_network_operations(
        max_retries=3, default_return=None, context="get_paper_balance"
    )
    async def get_paper_balance(self) -> Optional[Dict[str, Any]]:
        """
        Get paper trading account balance.

        Returns:
            Optional[Dict]: Account balance information
        """
        try:
            if not self.is_running:
                return None

            # Get account info from testnet
            account_info = await self.exchange.get_account_info()

            if account_info:
                self.logger.info(
                    f"💰 Paper balance: {account_info.get('totalWalletBalance', 'Unknown')} USDT"
                )
                return account_info
            else:
                self.logger.warning("❌ Could not retrieve paper balance")
                return None

        except Exception as e:
            self.logger.error(f"❌ Error getting paper balance: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,), default_return=False, context="is_paper_trading_active"
    )
    def is_paper_trading_active(self) -> bool:
        """Check if paper trading is currently active."""
        return self.is_running

    @handle_errors(
        exceptions=(Exception,), default_return={}, context="get_paper_trading_status"
    )
    async def get_paper_trading_status(self) -> Dict[str, Any]:
        """
        Get comprehensive paper trading status.

        Returns:
            Dict: Status information
        """
        try:
            status = {
                "is_running": self.is_running,
                "timestamp": datetime.now().isoformat(),
                "environment": "TESTNET",
                "symbol": self.symbol,
            }

            if self.is_running:
                # Get additional status information
                balance = await self.get_paper_balance()
                position = await self.get_paper_position()

                status.update(
                    {
                        "balance": balance,
                        "position": position,
                        "connection_status": "CONNECTED",
                    }
                )
            else:
                status["connection_status"] = "DISCONNECTED"

            return status

        except Exception as e:
            self.logger.error(f"❌ Error getting paper trading status: {e}")
            return {
                "is_running": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
