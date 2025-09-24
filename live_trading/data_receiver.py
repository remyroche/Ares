"""
Data Receiver

Handles market data streaming and processing from exchanges.
Provides real-time market data for trading decisions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass

from src.interfaces.base_interfaces import MarketData
from ..exchange.base_exchange import BaseExchange


@dataclass
class DataSubscription:
    """Data subscription configuration"""
    symbol: str
    interval: str
    callback: Optional[Callable] = None
    is_active: bool = False


class DataReceiver:
    """
    Handles market data streaming from exchanges.

    This class provides:
    - Real-time market data streaming
    - Historical data retrieval
    - Data processing and filtering
    - Subscription management
    """

    def __init__(self, exchange: BaseExchange, symbols: List[str]):
        self.exchange = exchange
        self.symbols = symbols
        self.logger = logging.getLogger(__name__)

        # Data storage
        self.market_data: Dict[str, List[MarketData]] = {}
        self.subscriptions: Dict[str, DataSubscription] = {}

        # Background tasks
        self.stream_task = None
        self.is_running = False

        # Event callbacks
        self.on_data_update: Optional[Callable] = None

        # Configuration
        self.max_data_points = 1000  # Max data points to keep per symbol
        self.update_interval = 1.0  # seconds between updates

    async def start(self) -> None:
        """Start data streaming."""
        try:
            if self.is_running:
                self.logger.warning("DataReceiver is already running")
                return

            self.logger.info("Starting DataReceiver...")

            # Initialize data storage
            for symbol in self.symbols:
                self.market_data[symbol] = []

            # Start streaming task
            self.stream_task = asyncio.create_task(self._stream_data())

            self.is_running = True
            self.logger.info("✅ DataReceiver started successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to start DataReceiver: {e}")
            raise

    async def stop(self) -> None:
        """Stop data streaming."""
        try:
            self.logger.info("Stopping DataReceiver...")

            self.is_running = False

            if self.stream_task:
                self.stream_task.cancel()
                try:
                    await self.stream_task
                except asyncio.CancelledError:
                    pass

            self.logger.info("✅ DataReceiver stopped successfully")

        except Exception as e:
            self.logger.error(f"❌ Error stopping DataReceiver: {e}")

    async def subscribe(self, symbol: str, interval: str = "1m", callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to market data for a symbol.

        Args:
            symbol: Trading symbol
            interval: Data interval
            callback: Optional callback for data updates

        Returns:
            True if subscription successful
        """
        try:
            subscription_id = f"{symbol}_{interval}"

            if subscription_id in self.subscriptions:
                self.logger.warning(f"Subscription already exists: {subscription_id}")
                return True

            subscription = DataSubscription(
                symbol=symbol,
                interval=interval,
                callback=callback,
                is_active=True
            )

            self.subscriptions[subscription_id] = subscription
            self.logger.info(f"✅ Subscribed to {symbol} ({interval})")

            return True

        except Exception as e:
            self.logger.error(f"❌ Error subscribing to {symbol}: {e}")
            return False

    async def unsubscribe(self, symbol: str, interval: str = "1m") -> bool:
        """
        Unsubscribe from market data for a symbol.

        Args:
            symbol: Trading symbol
            interval: Data interval

        Returns:
            True if unsubscription successful
        """
        try:
            subscription_id = f"{symbol}_{interval}"

            if subscription_id not in self.subscriptions:
                self.logger.warning(f"Subscription not found: {subscription_id}")
                return True

            self.subscriptions[subscription_id].is_active = False
            del self.subscriptions[subscription_id]

            self.logger.info(f"✅ Unsubscribed from {symbol} ({interval})")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error unsubscribing from {symbol}: {e}")
            return False

    async def get_latest_data(self, symbol: str, limit: int = 1) -> List[MarketData]:
        """
        Get the latest market data for a symbol.

        Args:
            symbol: Trading symbol
            limit: Number of data points to return

        Returns:
            List of MarketData objects
        """
        try:
            if symbol not in self.market_data:
                return []

            data = self.market_data[symbol]
            return data[-limit:] if data else []

        except Exception as e:
            self.logger.error(f"❌ Error getting latest data for {symbol}: {e}")
            return []

    async def get_historical_data(self, symbol: str, interval: str = "1m",
                                 limit: int = 100, start_time: Optional[datetime] = None) -> List[MarketData]:
        """
        Get historical market data for a symbol.

        Args:
            symbol: Trading symbol
            interval: Data interval
            limit: Number of data points
            start_time: Optional start time filter

        Returns:
            List of MarketData objects
        """
        try:
            # Get data from exchange
            data = await self.exchange.get_klines(symbol, interval, limit)

            # Filter by start time if provided
            if start_time and data:
                data = [d for d in data if d.timestamp >= start_time]

            # Update local storage
            if symbol not in self.market_data:
                self.market_data[symbol] = []

            self.market_data[symbol].extend(data)

            # Limit stored data points
            if len(self.market_data[symbol]) > self.max_data_points:
                self.market_data[symbol] = self.market_data[symbol][-self.max_data_points:]

            return data

        except Exception as e:
            self.logger.error(f"❌ Error getting historical data for {symbol}: {e}")
            return []

    async def get_aggregated_data(self, symbol: str, target_interval: str = "5m",
                                 source_interval: str = "1m", limit: int = 100) -> List[MarketData]:
        """
        Get aggregated market data for a symbol.

        Args:
            symbol: Trading symbol
            target_interval: Target interval for aggregation
            source_interval: Source interval for data
            limit: Number of data points

        Returns:
            List of aggregated MarketData objects
        """
        try:
            # Get raw data
            raw_data = await self.get_historical_data(symbol, source_interval, limit * 5)

            if not raw_data:
                return []

            # Aggregate data
            aggregated = self._aggregate_data(raw_data, target_interval)

            return aggregated

        except Exception as e:
            self.logger.error(f"❌ Error getting aggregated data for {symbol}: {e}")
            return []

    def _aggregate_data(self, data: List[MarketData], target_interval: str) -> List[MarketData]:
        """
        Aggregate market data to a target interval.

        Args:
            data: Raw market data
            target_interval: Target interval

        Returns:
            Aggregated market data
        """
        try:
            if not data:
                return []

            # Group data by target interval
            aggregated = []
            current_group = []
            target_minutes = self._interval_to_minutes(target_interval)

            for data_point in sorted(data, key=lambda x: x.timestamp):
                current_group.append(data_point)

                # Check if we should aggregate this group
                if len(current_group) >= target_minutes:
                    aggregated.append(self._create_aggregated_point(current_group, target_interval))
                    current_group = []

            # Handle remaining points
            if current_group:
                aggregated.append(self._create_aggregated_point(current_group, target_interval))

            return aggregated

        except Exception as e:
            self.logger.error(f"❌ Error aggregating data: {e}")
            return data  # Return original data on error

    def _create_aggregated_point(self, data_group: List[MarketData], interval: str) -> MarketData:
        """
        Create an aggregated data point from a group of data points.

        Args:
            data_group: Group of data points to aggregate
            interval: Target interval

        Returns:
            Aggregated MarketData object
        """
        try:
            if not data_group:
                raise ValueError("Empty data group")

            # Get first and last points
            first = data_group[0]
            last = data_group[-1]

            # Calculate aggregated values
            opens = [d.open for d in data_group]
            highs = [d.high for d in data_group]
            lows = [d.low for d in data_group]
            closes = [d.close for d in data_group]
            volumes = [d.volume for d in data_group]

            return MarketData(
                symbol=first.symbol,
                timestamp=first.timestamp,  # Use first timestamp
                open=opens[0],  # First open
                high=max(highs),  # Highest high
                low=min(lows),    # Lowest low
                close=closes[-1], # Last close
                volume=sum(volumes), # Total volume
                interval=interval
            )

        except Exception as e:
            self.logger.error(f"❌ Error creating aggregated point: {e}")
            return data_group[0]  # Return first point on error

    def _interval_to_minutes(self, interval: str) -> int:
        """
        Convert interval string to minutes.

        Args:
            interval: Interval string (e.g., "1m", "5m", "1h")

        Returns:
            Interval in minutes
        """
        try:
            if interval.endswith('m'):
                return int(interval[:-1])
            elif interval.endswith('h'):
                return int(interval[:-1]) * 60
            elif interval.endswith('d'):
                return int(interval[:-1]) * 24 * 60
            else:
                return 1  # Default to 1 minute
        except Exception:
            return 1

    async def _stream_data(self) -> None:
        """Background task for streaming data."""
        while self.is_running:
            try:
                # Update data for all subscriptions
                for subscription_id, subscription in self.subscriptions.items():
                    if subscription.is_active:
                        await self._update_subscription_data(subscription)

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in data streaming: {e}")
                await asyncio.sleep(5)

    async def _update_subscription_data(self, subscription: DataSubscription) -> None:
        """
        Update data for a subscription.

        Args:
            subscription: The data subscription
        """
        try:
            # Get latest data
            latest_data = await self.get_historical_data(
                subscription.symbol,
                subscription.interval,
                limit=1
            )

            if latest_data:
                data_point = latest_data[-1]

                # Update local storage
                if subscription.symbol not in self.market_data:
                    self.market_data[subscription.symbol] = []

                # Add data point if it's new
                if not self.market_data[subscription.symbol] or \
                   self.market_data[subscription.symbol][-1].timestamp != data_point.timestamp:
                    self.market_data[subscription.symbol].append(data_point)

                    # Limit stored data points
                    if len(self.market_data[subscription.symbol]) > self.max_data_points:
                        self.market_data[subscription.symbol] = \
                            self.market_data[subscription.symbol][-self.max_data_points:]

                    # Call callback if provided
                    if subscription.callback:
                        await subscription.callback(data_point)

                    # Call global callback
                    if self.on_data_update:
                        await self.on_data_update(data_point)

        except Exception as e:
            self.logger.error(f"❌ Error updating subscription data for {subscription.symbol}: {e}")

    # Additional utility methods
    async def get_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price or None if not available
        """
        try:
            latest_data = await self.get_latest_data(symbol, 1)
            if latest_data:
                return latest_data[0].close
            return None

        except Exception as e:
            self.logger.error(f"❌ Error getting price for {symbol}: {e}")
            return None

    async def get_spread(self, symbol: str) -> Optional[float]:
        """
        Get current spread for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current spread or None if not available
        """
        try:
            latest_data = await self.get_latest_data(symbol, 1)
            if latest_data:
                data = latest_data[0]
                return data.high - data.low
            return None

        except Exception as e:
            self.logger.error(f"❌ Error getting spread for {symbol}: {e}")
            return None

    async def get_volume(self, symbol: str) -> Optional[float]:
        """
        Get current volume for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current volume or None if not available
        """
        try:
            latest_data = await self.get_latest_data(symbol, 1)
            if latest_data:
                return latest_data[0].volume
            return None

        except Exception as e:
            self.logger.error(f"❌ Error getting volume for {symbol}: {e}")
            return None

    # Configuration methods
    def set_data_update_callback(self, callback: Callable):
        """Set callback for data updates."""
        self.on_data_update = callback

    def set_max_data_points(self, max_points: int):
        """Set maximum number of data points to store."""
        self.max_data_points = max_points

    def set_update_interval(self, interval: float):
        """Set update interval in seconds."""
        self.update_interval = interval