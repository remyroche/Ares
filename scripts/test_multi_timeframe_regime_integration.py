#!/usr/bin/env python3
"""
Test script for Multi-Timeframe Regime Integration

This script demonstrates the integration of HMM regime classification with the multi-timeframe system:

1. Training the HMM classifier on 1h data
2. Classifying regimes using 1h data only
3. Propagating regime information to other timeframes
4. Getting regime-specific TP/SL optimization
5. Testing the integration with different timeframes

Usage:
    python scripts/test_multi_timeframe_regime_integration.py --symbol ETHUSDT
    python scripts/test_multi_timeframe_regime_integration.py --symbol BTCUSDT --train-hmm
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyst.multi_timeframe_regime_integration import (
    MultiTimeframeRegimeIntegration,
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


class MultiTimeframeRegimeIntegrationTester:
    """
    Test and demonstrate the multi-timeframe regime integration system.
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

        # Initialize multi-timeframe regime integration
        self.regime_integration = MultiTimeframeRegimeIntegration(config)

        # Test timeframes
        self.test_timeframes = ["1m", "5m", "15m", "1h"]

        terminal_log(
            f"🚀 Initialized MultiTimeframeRegimeIntegrationTester for {symbol}",
        )

    async def initialize(self) -> bool:
        """
        Initialize the tester components.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            terminal_log("Initializing Multi-Timeframe Regime Integration Tester...")

            # Initialize regime integration
            if not await self.regime_integration.initialize():
                terminal_log("❌ Failed to initialize regime integration", "ERROR")
                return False

            terminal_log(
                "✅ Multi-Timeframe Regime Integration Tester initialized successfully",
            )
            return True

        except Exception as e:
            terminal_log(f"❌ Failed to initialize tester: {e}", "ERROR")
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
                    limit=1000,  # Get last 1000 candles
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

    async def train_hmm_classifier(self, data_1h: pd.DataFrame) -> bool:
        """
        Train the HMM classifier with 1h data.

        Args:
            data_1h: 1-hour timeframe data

        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            terminal_log("🎓 Training HMM regime classifier...")

            if data_1h is None or data_1h.empty:
                terminal_log("❌ No 1h data available for training", "ERROR")
                return False

            success = await self.regime_integration.train_hmm_classifier(data_1h)

            if success:
                terminal_log("✅ HMM regime classifier trained successfully")
            else:
                terminal_log("❌ Failed to train HMM regime classifier", "ERROR")

            return success

        except Exception as e:
            terminal_log(f"❌ Error training HMM classifier: {e}", "ERROR")
            return False

    async def test_regime_classification(self, data_1h: pd.DataFrame) -> None:
        """
        Test regime classification using 1h data.

        Args:
            data_1h: 1-hour timeframe data
        """
        try:
            terminal_log("🔍 Testing regime classification with 1h data...")

            if data_1h is None or data_1h.empty:
                terminal_log("❌ No 1h data available for testing", "ERROR")
                return

            # Get current regime
            regime, confidence, info = await self.regime_integration.classify_regime_1h(
                data_1h,
            )

            terminal_log(f"📊 Current regime: {regime}")
            terminal_log(f"📊 Confidence: {confidence:.2f}")
            terminal_log(f"📊 Additional info: {info}")

        except Exception as e:
            terminal_log(f"❌ Error in regime classification test: {e}", "ERROR")

    async def test_regime_propagation(self, test_data: dict) -> None:
        """
        Test regime propagation to different timeframes.

        Args:
            test_data: Dictionary with timeframe -> DataFrame mapping
        """
        try:
            terminal_log("🔄 Testing regime propagation to different timeframes...")

            if "1h" not in test_data:
                terminal_log("❌ No 1h data available for regime propagation", "ERROR")
                return

            data_1h = test_data["1h"]

            for timeframe in self.test_timeframes:
                if timeframe in test_data:
                    terminal_log(f"📊 Testing regime for {timeframe}...")

                    regime_info = (
                        await self.regime_integration.get_regime_for_timeframe(
                            timeframe=timeframe,
                            current_data=test_data[timeframe],
                            data_1h=data_1h,
                        )
                    )

                    terminal_log(f"   Regime: {regime_info['regime']}")
                    terminal_log(f"   Confidence: {regime_info['confidence']:.2f}")
                    terminal_log(f"   Timeframe: {regime_info['timeframe']}")
                    terminal_log(
                        f"   Strategic timeframe: {regime_info['strategic_timeframe']}",
                    )

                    if "timeframe_adjustment" in regime_info:
                        adjustment = regime_info["timeframe_adjustment"]
                        terminal_log(
                            f"   Volatility multiplier: {adjustment.get('volatility_multiplier', 1.0)}",
                        )
                        terminal_log(
                            f"   Momentum threshold: {adjustment.get('momentum_threshold', 0.5)}",
                        )

                else:
                    terminal_log(f"⚠️ No data available for {timeframe}", "WARNING")

        except Exception as e:
            terminal_log(f"❌ Error in regime propagation test: {e}", "ERROR")

    async def test_regime_specific_optimization(self, test_data: dict) -> None:
        """
        Test regime-specific TP/SL optimization.

        Args:
            test_data: Dictionary with timeframe -> DataFrame mapping
        """
        try:
            terminal_log("🎯 Testing regime-specific TP/SL optimization...")

            if "1h" not in test_data:
                terminal_log("❌ No 1h data available for optimization", "ERROR")
                return

            data_1h = test_data["1h"]

            for timeframe in self.test_timeframes:
                if timeframe in test_data:
                    terminal_log(f"🎯 Testing optimization for {timeframe}...")

                    optimization_params = await self.regime_integration.get_regime_specific_optimization(
                        timeframe=timeframe,
                        current_data=test_data[timeframe],
                        data_1h=data_1h,
                        historical_data=data_1h,  # Use 1h data as historical data for simplicity
                    )

                    terminal_log(f"   Regime: {optimization_params['regime']}")
                    terminal_log(
                        f"   Target %: {optimization_params.get('target_pct', 0):.3f}",
                    )
                    terminal_log(
                        f"   Stop %: {optimization_params.get('stop_pct', 0):.3f}",
                    )
                    terminal_log(
                        f"   Risk/Reward: {optimization_params.get('risk_reward_ratio', 0):.2f}",
                    )
                    terminal_log(
                        f"   Success rate: {optimization_params.get('success_rate', 0):.2f}%",
                    )
                    terminal_log(
                        f"   Avg duration: {optimization_params.get('avg_duration_minutes', 0):.1f} min",
                    )

                else:
                    terminal_log(f"⚠️ No data available for {timeframe}", "WARNING")

        except Exception as e:
            terminal_log(f"❌ Error in regime-specific optimization test: {e}", "ERROR")

    async def get_integration_statistics(self) -> None:
        """
        Get and display integration statistics.
        """
        try:
            terminal_log("📈 Getting integration statistics...")

            stats = self.regime_integration.get_integration_statistics()

            terminal_log("📊 Integration Statistics:")
            terminal_log(f"   Current regime: {stats['current_regime']}")
            terminal_log(f"   Regime confidence: {stats['regime_confidence']:.2f}")
            terminal_log(f"   HMM trained: {stats['hmm_trained']}")
            terminal_log(f"   Strategic timeframe: {stats['strategic_timeframe']}")
            terminal_log(f"   Active timeframes: {stats['active_timeframes']}")
            terminal_log(
                f"   Cache duration: {stats['regime_cache_duration_minutes']:.1f} minutes",
            )

            if stats["last_regime_update"]:
                terminal_log(f"   Last regime update: {stats['last_regime_update']}")

            # Display TP/SL optimizer stats
            tpsl_stats = stats["regime_tpsl_optimizer_stats"]
            terminal_log(
                f"   Optimized regimes: {tpsl_stats.get('optimized_regimes', [])}",
            )
            terminal_log(
                f"   Total optimizations: {tpsl_stats.get('total_optimizations', 0)}",
            )

        except Exception as e:
            terminal_log(f"❌ Error getting integration statistics: {e}", "ERROR")

    async def run_full_test(self, train_hmm: bool = False) -> None:
        """
        Run the full test suite.

        Args:
            train_hmm: Whether to train the HMM classifier
        """
        try:
            terminal_log("🚀 Starting full multi-timeframe regime integration test...")

            # Initialize
            if not await self.initialize():
                return

            # Load test data
            test_data = await self.load_test_data()

            if not test_data:
                terminal_log("❌ No test data available", "ERROR")
                return

            # Train HMM classifier if requested
            if train_hmm and "1h" in test_data:
                await self.train_hmm_classifier(test_data["1h"])

            # Test regime classification
            if "1h" in test_data:
                await self.test_regime_classification(test_data["1h"])

            # Test regime propagation
            await self.test_regime_propagation(test_data)

            # Test regime-specific optimization
            await self.test_regime_specific_optimization(test_data)

            # Get statistics
            await self.get_integration_statistics()

            terminal_log("✅ Full test completed successfully!")

        except Exception as e:
            terminal_log(f"❌ Error in full test: {e}", "ERROR")


async def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(
        description="Test Multi-Timeframe Regime Integration",
    )
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--train-hmm", action="store_true", help="Train HMM classifier")

    args = parser.parse_args()

    terminal_log("=" * 60)
    terminal_log("🧪 MULTI-TIMEFRAME REGIME INTEGRATION TEST")
    terminal_log("=" * 60)
    terminal_log(f"Symbol: {args.symbol}")
    terminal_log(f"Train HMM: {args.train_hmm}")
    terminal_log("=" * 60)

    # Create tester
    tester = MultiTimeframeRegimeIntegrationTester(args.symbol, CONFIG)

    # Run full test
    await tester.run_full_test(train_hmm=args.train_hmm)

    terminal_log("=" * 60)
    terminal_log("🏁 Test completed!")
    terminal_log("=" * 60)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
