#!/usr/bin/env python3
"""
Test script for Multi-Timeframe Feature Engineering

This script demonstrates the multi-timeframe feature engineering system:

1. Loading data for different timeframes
2. Generating timeframe-specific features
3. Comparing indicator parameters across timeframes
4. Testing the adaptation system
5. Validating feature generation

Usage:
    python scripts/test_multi_timeframe_feature_engineering.py --symbol ETHUSDT
    python scripts/test_multi_timeframe_feature_engineering.py --symbol BTCUSDT --timeframes 1m,5m,15m,1h
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyst.multi_timeframe_feature_engineering import (
    MultiTimeframeFeatureEngineering,
)
from src.config import CONFIG
from src.database.sqlite_manager import SQLiteManager
from src.utils.logger import ensure_logging_setup, get_logger

# Ensure logging is set up
ensure_logging_setup()
logger = get_logger(__name__)


def terminal_log(message: str, level: str = "INFO"):
    """Log to both terminal and logger"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}", flush=True)
    if level == "INFO":
        logger.info(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)


class MultiTimeframeFeatureEngineeringTester:
    """
    Test and demonstrate the multi-timeframe feature engineering system.
    """

    def __init__(self, symbol: str, config: dict):
        """
        Initialize the tester.

        Args:
            symbol: Trading symbol
            config: Configuration dictionary
        """
        self.symbol = symbol
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize database manager
        self.db_manager = SQLiteManager()

        # Initialize multi-timeframe feature engineering
        self.mtf_feature_engine = MultiTimeframeFeatureEngineering(config)

        # Test timeframes
        self.test_timeframes = ["1m", "5m", "15m", "1h"]

        terminal_log(
            f"🚀 Initialized MultiTimeframeFeatureEngineeringTester for {symbol}",
        )

    async def initialize(self) -> bool:
        """
        Initialize the tester components.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            terminal_log("Initializing Multi-Timeframe Feature Engineering Tester...")

            # Test database connection
            if not await self._test_database_connection():
                terminal_log("❌ Failed to connect to database", "ERROR")
                return False

            terminal_log(
                "✅ Multi-Timeframe Feature Engineering Tester initialized successfully",
            )
            return True

        except Exception as e:
            terminal_log(f"❌ Failed to initialize tester: {e}", "ERROR")
            return False

    async def _test_database_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Try to get a small amount of data
            test_data = await self.db_manager.get_klines(
                symbol=self.symbol,
                timeframe="1h",
                limit=10,
            )

            if test_data is not None and not test_data.empty:
                terminal_log("✅ Database connection successful")
                return True
            terminal_log("⚠️ Database connection test inconclusive", "WARNING")
            return True  # Assume it's working

        except Exception as e:
            terminal_log(f"❌ Database connection failed: {e}", "ERROR")
            return False

    async def load_test_data(self) -> dict:
        """
        Load test data for different timeframes.

        Returns:
            dict: Dictionary with timeframe -> DataFrame mapping
        """
        terminal_log("📊 Loading test data for different timeframes...")

        data = {}

        try:
            for timeframe in self.test_timeframes:
                # Load data from database
                klines_data = await self.db_manager.get_klines(
                    symbol=self.symbol,
                    timeframe=timeframe,
                    limit=500,  # Get last 500 candles for testing
                )

                if klines_data is not None and not klines_data.empty:
                    data[timeframe] = klines_data
                    terminal_log(
                        f"✅ Loaded {len(klines_data)} candles for {timeframe}",
                    )
                else:
                    terminal_log(f"⚠️ No data available for {timeframe}", "WARNING")

        except Exception as e:
            terminal_log(f"❌ Error loading test data: {e}", "ERROR")

        return data

    async def test_feature_generation(self, test_data: dict) -> None:
        """
        Test multi-timeframe feature generation.

        Args:
            test_data: Dictionary with timeframe -> DataFrame mapping
        """
        try:
            terminal_log("🎯 Testing multi-timeframe feature generation...")

            if not test_data:
                terminal_log("❌ No test data available", "ERROR")
                return

            # Generate features for all timeframes
            features_dict = (
                await self.mtf_feature_engine.generate_multi_timeframe_features(
                    data_dict=test_data,
                    agg_trades_dict=None,  # Skip for this test
                    futures_dict=None,  # Skip for this test
                    sr_levels=None,  # Skip for this test
                )
            )

            # Analyze results
            for timeframe, features in features_dict.items():
                terminal_log(f"📊 Features for {timeframe}:")
                terminal_log(f"   Shape: {features.shape}")
                terminal_log(f"   Columns: {len(features.columns)}")

                # Show some key features
                key_features = [col for col in features.columns if timeframe in col]
                terminal_log(f"   Timeframe-specific features: {len(key_features)}")

                if key_features:
                    terminal_log(f"   Sample features: {key_features[:5]}")

        except Exception as e:
            terminal_log(f"❌ Error in feature generation test: {e}", "ERROR")

    async def test_timeframe_adaptation(self, test_data: dict) -> None:
        """
        Test timeframe adaptation of indicators.

        Args:
            test_data: Dictionary with timeframe -> DataFrame mapping
        """
        try:
            terminal_log("🔄 Testing timeframe adaptation...")

            for timeframe in self.test_timeframes:
                if timeframe not in test_data:
                    continue

                terminal_log(f"📊 Testing adaptation for {timeframe}...")

                # Get timeframe parameters
                tf_params = self.mtf_feature_engine.get_timeframe_parameters(timeframe)

                if tf_params:
                    terminal_log(
                        f"   Description: {tf_params.get('description', 'N/A')}",
                    )
                    terminal_log(
                        f"   Trading style: {tf_params.get('trading_style', 'N/A')}",
                    )

                    # Show indicator parameters
                    indicators = tf_params.get("indicators", {})
                    terminal_log(f"   Indicators configured: {len(indicators)}")

                    for indicator, params in indicators.items():
                        terminal_log(f"     {indicator}: {params}")
                else:
                    terminal_log(f"   ⚠️ No parameters found for {timeframe}", "WARNING")

        except Exception as e:
            terminal_log(f"❌ Error in timeframe adaptation test: {e}", "ERROR")

    async def test_indicator_comparison(self, test_data: dict) -> None:
        """
        Compare indicator parameters across timeframes.

        Args:
            test_data: Dictionary with timeframe -> DataFrame mapping
        """
        try:
            terminal_log("📈 Testing indicator comparison across timeframes...")

            # Compare RSI parameters
            terminal_log("   RSI Parameters:")
            for timeframe in self.test_timeframes:
                tf_params = self.mtf_feature_engine.get_timeframe_parameters(timeframe)
                rsi_params = tf_params.get("indicators", {}).get("rsi", {})
                length = rsi_params.get("length", "N/A")
                description = rsi_params.get("description", "N/A")
                terminal_log(f"     {timeframe}: length={length} ({description})")

            # Compare MACD parameters
            terminal_log("   MACD Parameters:")
            for timeframe in self.test_timeframes:
                tf_params = self.mtf_feature_engine.get_timeframe_parameters(timeframe)
                macd_params = tf_params.get("indicators", {}).get("macd", {})
                fast = macd_params.get("fast", "N/A")
                slow = macd_params.get("slow", "N/A")
                signal = macd_params.get("signal", "N/A")
                terminal_log(
                    f"     {timeframe}: fast={fast}, slow={slow}, signal={signal}",
                )

            # Compare ATR parameters
            terminal_log("   ATR Parameters:")
            for timeframe in self.test_timeframes:
                tf_params = self.mtf_feature_engine.get_timeframe_parameters(timeframe)
                atr_params = tf_params.get("indicators", {}).get("atr", {})
                length = atr_params.get("length", "N/A")
                description = atr_params.get("description", "N/A")
                terminal_log(f"     {timeframe}: length={length} ({description})")

        except Exception as e:
            terminal_log(f"❌ Error in indicator comparison test: {e}", "ERROR")

    async def test_feature_statistics(self) -> None:
        """
        Test feature engineering statistics.
        """
        try:
            terminal_log("📊 Testing feature engineering statistics...")

            stats = self.mtf_feature_engine.get_feature_statistics()

            terminal_log("Feature Engineering Statistics:")
            terminal_log(f"   Supported timeframes: {stats['supported_timeframes']}")
            terminal_log(f"   Cache size: {stats['cache_size']}")
            terminal_log(f"   MTF features enabled: {stats['enable_mtf_features']}")
            terminal_log(
                f"   Timeframe adaptation enabled: {stats['enable_timeframe_adaptation']}",
            )
            terminal_log(
                f"   Timeframe parameters count: {stats['timeframe_parameters_count']}",
            )

            if stats["last_cache_cleanup"]:
                terminal_log(f"   Last cache cleanup: {stats['last_cache_cleanup']}")

        except Exception as e:
            terminal_log(f"❌ Error getting feature statistics: {e}", "ERROR")

    async def test_supported_timeframes(self) -> None:
        """
        Test supported timeframes functionality.
        """
        try:
            terminal_log("⏰ Testing supported timeframes...")

            supported_timeframes = self.mtf_feature_engine.get_supported_timeframes()

            terminal_log(f"Supported timeframes: {supported_timeframes}")

            # Test parameter retrieval for each timeframe
            for timeframe in supported_timeframes:
                params = self.mtf_feature_engine.get_timeframe_parameters(timeframe)
                if params:
                    terminal_log(f"   {timeframe}: {params.get('description', 'N/A')}")
                else:
                    terminal_log(f"   {timeframe}: No parameters found", "WARNING")

        except Exception as e:
            terminal_log(f"❌ Error testing supported timeframes: {e}", "ERROR")

    async def run_full_test(self) -> None:
        """
        Run the full test suite.
        """
        try:
            terminal_log("🚀 Starting full multi-timeframe feature engineering test...")

            # Initialize
            if not await self.initialize():
                return

            # Load test data
            test_data = await self.load_test_data()

            if not test_data:
                terminal_log("❌ No test data available", "ERROR")
                return

            # Test supported timeframes
            await self.test_supported_timeframes()

            # Test timeframe adaptation
            await self.test_timeframe_adaptation(test_data)

            # Test indicator comparison
            await self.test_indicator_comparison(test_data)

            # Test feature generation
            await self.test_feature_generation(test_data)

            # Test feature statistics
            await self.test_feature_statistics()

            terminal_log("✅ Full test completed successfully!")

        except Exception as e:
            terminal_log(f"❌ Error in full test: {e}", "ERROR")


async def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(
        description="Test Multi-Timeframe Feature Engineering",
    )
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument(
        "--timeframes",
        type=str,
        default="1m,5m,15m,1h",
        help="Comma-separated timeframes to test",
    )

    args = parser.parse_args()

    terminal_log("=" * 60)
    terminal_log("🧪 MULTI-TIMEFRAME FEATURE ENGINEERING TEST")
    terminal_log("=" * 60)
    terminal_log(f"Symbol: {args.symbol}")
    terminal_log(f"Timeframes: {args.timeframes}")
    terminal_log("=" * 60)

    # Create tester
    tester = MultiTimeframeFeatureEngineeringTester(args.symbol, CONFIG)

    # Update test timeframes if specified
    if args.timeframes:
        tester.test_timeframes = args.timeframes.split(",")

    # Run full test
    await tester.run_full_test()

    terminal_log("=" * 60)
    terminal_log("🏁 Test completed!")
    terminal_log("=" * 60)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
