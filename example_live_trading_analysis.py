#!/usr/bin/env python3
"""
Example: Live Trading Analysis with ML Model

This script demonstrates how to use the LiveDataCollector to fetch market data
every 30 seconds and analyze it with your trained ML model for live trading.
"""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.trading.data.live_data_collector import (
    LiveDataCollector,
    LiveDataConfig,
    CollectionMode,
    DataQuality,
    CollectionInterval,
    start_live_collection
)
from src.utils.logger import system_logger

logger = system_logger.getChild('LiveTradingExample')

class LiveTradingAnalyzer:
    """
    Example trading analyzer that processes live data with ML predictions.
    """

    def __init__(self, symbol: str = "ETH", ml_model_path: str = None):
        self.symbol = symbol
        self.ml_model_path = ml_model_path
        self.collector = None

        # Trading state
        self.position = None  # None, 'long', 'short'
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0

        # Performance tracking
        self.trades = []
        self.total_pnl = 0.0

    async def initialize(self):
        """Initialize the live data collector."""
        print(f"🚀 Initializing live trading analyzer for {self.symbol}")

        # Choose collection interval (FAST=15s, STANDARD=30s)
        interval = CollectionInterval.STANDARD  # Use 30-second intervals

        # Create live data collector with ML integration
        self.collector = await start_live_collection(
            symbol=self.symbol,
            exchange="binance",
            interval=interval,
            ml_model_path=self.ml_model_path,
            data_callback=self.on_new_data
        )

        # Add error callback
        self.collector.add_error_callback(self.on_error)

        print("✅ Live trading analyzer initialized")

    async def on_new_data(self, data_point):
        """Process new data point with ML analysis."""
        try:
            # Extract data
            raw_data = data_point.raw_data
            processed_data = data_point.processed_data or {}
            ml_predictions = data_point.ml_predictions or {}

            # Log the data point
            print(f"\n📊 {data_point.timestamp.strftime('%H:%M:%S')} | "
                  f"Price: ${raw_data['close']:.2f} | "
                  f"Volume: {raw_data['volume']:.4f}")

            # Log processed features
            if processed_data:
                returns = processed_data.get('returns', 0) * 100
                volatility = processed_data.get('volatility', 0) * 100
                print(f"   📈 Returns: {returns:.4f}% | Volatility: {volatility:.4f}%")

            # Log ML predictions
            if ml_predictions:
                prediction = ml_predictions.get('prediction', 'N/A')
                confidence = ml_predictions.get('confidence', 0) * 100
                print(f"   🤖 ML Prediction: {prediction} | Confidence: {confidence:.1f}%")

            # Make trading decision based on ML prediction
            await self.make_trading_decision(data_point)

            # Log current position
            if self.position:
                pnl = self.calculate_current_pnl(raw_data['close'])
                print(f"   📊 Position: {self.position.upper()} | PnL: ${pnl:.2f}")
            else:
                print("   📊 Position: NONE (waiting for signal)")

        except Exception as e:
            logger.error(f"❌ Error processing data point: {e}")

    async def make_trading_decision(self, data_point):
        """Make trading decisions based on ML predictions and market data."""
        try:
            ml_predictions = data_point.ml_predictions
            if not ml_predictions:
                return

            prediction = ml_predictions.get('prediction')
            confidence = ml_predictions.get('confidence', 0)
            current_price = data_point.raw_data['close']

            # Define trading thresholds
            min_confidence = 0.7  # 70% confidence required
            stop_loss_pct = 0.02  # 2% stop loss
            take_profit_pct = 0.04  # 4% take profit

            # Check for entry signals
            if self.position is None:
                if prediction == 1 and confidence > min_confidence:  # Bullish signal
                    await self.enter_position('long', current_price, stop_loss_pct, take_profit_pct)
                elif prediction == 0 and confidence > min_confidence:  # Bearish signal
                    await self.enter_position('short', current_price, stop_loss_pct, take_profit_pct)

            # Check for exit signals (if in position)
            elif self.position == 'long':
                if current_price <= self.stop_loss or current_price >= self.take_profit:
                    await self.exit_position(current_price, "Stop loss/take profit hit")
                elif prediction == 0 and confidence > min_confidence:  # Bearish reversal
                    await self.exit_position(current_price, "Reversal signal")

            elif self.position == 'short':
                if current_price >= self.stop_loss or current_price <= self.take_profit:
                    await self.exit_position(current_price, "Stop loss/take profit hit")
                elif prediction == 1 and confidence > min_confidence:  # Bullish reversal
                    await self.exit_position(current_price, "Reversal signal")

        except Exception as e:
            logger.error(f"❌ Error making trading decision: {e}")

    async def enter_position(self, side: str, price: float, sl_pct: float, tp_pct: float):
        """Enter a trading position."""
        self.position = side
        self.entry_price = price

        if side == 'long':
            self.stop_loss = price * (1 - sl_pct)
            self.take_profit = price * (1 + tp_pct)
        else:  # short
            self.stop_loss = price * (1 + sl_pct)
            self.take_profit = price * (1 - tp_pct)

        print(f"🚀 ENTERED {side.upper()} at ${price:.2f} | SL: ${self.stop_loss:.2f} | TP: ${self.take_profit:.2f}")

    async def exit_position(self, exit_price: float, reason: str):
        """Exit the current position."""
        if not self.position:
            return

        # Calculate P&L
        if self.position == 'long':
            pnl = exit_price - self.entry_price
        else:  # short
            pnl = self.entry_price - exit_price

        self.total_pnl += pnl

        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'side': self.position,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'reason': reason
        }
        self.trades.append(trade)

        print(f"🔚 EXITED {self.position.upper()} at ${exit_price:.2f} | "
              f"P&L: ${pnl:.2f} | Reason: {reason}")
        print(f"📊 Total P&L: ${self.total_pnl:.2f} | Total Trades: {len(self.trades)}")

        # Reset position
        self.position = None
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0

    def calculate_current_pnl(self, current_price: float) -> float:
        """Calculate current unrealized P&L."""
        if not self.position:
            return 0.0

        if self.position == 'long':
            return current_price - self.entry_price
        else:  # short
            return self.entry_price - current_price

    async def on_error(self, error: Exception):
        """Handle collection errors."""
        logger.error(f"❌ Data collection error: {error}")
        print(f"⚠️ Data collection error occurred: {error}")

    async def get_performance_summary(self):
        """Get trading performance summary."""
        if not self.trades:
            return "No trades executed yet"

        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]

        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

        return f"""
📊 Performance Summary:
   Total Trades: {len(self.trades)}
   Win Rate: {win_rate:.1f}%
   Total P&L: ${self.total_pnl:.2f}
   Average Win: ${avg_win:.2f}
   Average Loss: ${avg_loss:.2f}
   Profit Factor: {abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')):.2f}
"""

    async def shutdown(self):
        """Shutdown the analyzer and collector."""
        print("\n🛑 Shutting down live trading analyzer...")

        if self.collector:
            await self.collector.stop_collection()

        # Print final performance summary
        summary = await self.get_performance_summary()
        print(summary)

        print("✅ Shutdown complete")

async def main():
    """Main function to run the live trading analysis example."""
    print("🤖 Live Trading Analysis Example")
    print("=" * 50)

    # Configuration
    symbol = "ETH"
    ml_model_path = "models/your_trained_model.pkl"  # Update this path

    # Check if ML model exists
    if not Path(ml_model_path).exists():
        print(f"⚠️ ML model not found at {ml_model_path}")
        print("   The system will run without ML predictions for demonstration")
        ml_model_path = None

    # Create analyzer
    analyzer = LiveTradingAnalyzer(symbol=symbol, ml_model_path=ml_model_path)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n📡 Received signal {signum}, initiating shutdown...")
        asyncio.create_task(analyzer.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize and start
        await analyzer.initialize()

        # Get collector stats
        stats = analyzer.collector.get_stats()
        print(f"📈 Collection Stats: {stats}")

        print("\n🎯 Live analysis started! Press Ctrl+C to stop.")
        print("   - Data collection: Every 30 seconds")
        print("   - ML analysis: Real-time predictions")
        print("   - Trading signals: Based on ML confidence > 70%")
        print("   - Risk management: 2% stop loss, 4% take profit")

        # Keep running until interrupted
        while analyzer.collector and analyzer.collector.is_running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        await analyzer.shutdown()

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
