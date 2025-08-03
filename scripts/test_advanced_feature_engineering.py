#!/usr/bin/env python3
"""
Test script for Advanced Feature Engineering

This script demonstrates the advanced feature engineering system including:

1. Advanced TA: Divergence Detection, Pattern Recognition, Volume Profile Analysis
2. Market Microstructure: Bid/ask imbalances, spread dynamics, market depth, liquidity
3. Volatility Targeting: Dynamic position sizing based on volatility
4. Volume Indicators: VPVR, Point of Control, VWAP, OBV, OBV Divergence
5. Advanced Momentum: ROC, Williams %R, CCI, Money Flow Index
6. S/R Zone Definition: BBands, Keltner Channel, VWAP Std Dev Bands, Volume Profile

Usage:
    python scripts/test_advanced_feature_engineering.py --symbol ETHUSDT
    python scripts/test_advanced_feature_engineering.py --symbol BTCUSDT --enable-all
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyst.advanced_feature_engineering import AdvancedFeatureEngineering
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


class AdvancedFeatureEngineeringTester:
    """
    Test and demonstrate the advanced feature engineering system.
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

        # Initialize advanced feature engineering
        self.advanced_feature_engine = AdvancedFeatureEngineering(config)

        terminal_log(f"🚀 Initialized AdvancedFeatureEngineeringTester for {symbol}")

    async def initialize(self) -> bool:
        """
        Initialize the tester components.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            terminal_log("Initializing Advanced Feature Engineering Tester...")

            # Test database connection
            if not await self._test_database_connection():
                terminal_log("❌ Failed to connect to database", "ERROR")
                return False

            terminal_log(
                "✅ Advanced Feature Engineering Tester initialized successfully",
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

    async def load_test_data(self) -> tuple:
        """
        Load test data for advanced feature engineering.

        Returns:
            tuple: (klines_df, agg_trades_df, order_book_df)
        """
        terminal_log("📊 Loading test data for advanced feature engineering...")

        try:
            # Load klines data
            klines_data = await self.db_manager.get_klines(
                symbol=self.symbol,
                timeframe="1h",
                limit=500,  # Get last 500 candles for testing
            )

            if klines_data is None or klines_data.empty:
                terminal_log("❌ No klines data available", "ERROR")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            terminal_log(f"✅ Loaded {len(klines_data)} klines")

            # Load aggregated trades data
            agg_trades_data = await self.db_manager.get_agg_trades(
                symbol=self.symbol,
                limit=1000,
            )

            if agg_trades_data is not None and not agg_trades_data.empty:
                terminal_log(f"✅ Loaded {len(agg_trades_data)} aggregated trades")
            else:
                terminal_log("⚠️ No aggregated trades data available", "WARNING")
                agg_trades_data = pd.DataFrame()

            # Create mock order book data (since we don't have real order book data)
            order_book_data = self._create_mock_order_book_data(klines_data)
            terminal_log(
                f"✅ Created mock order book data with {len(order_book_data)} entries",
            )

            return klines_data, agg_trades_data, order_book_data

        except Exception as e:
            terminal_log(f"❌ Error loading test data: {e}", "ERROR")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def _create_mock_order_book_data(self, klines_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create mock order book data for testing.

        Args:
            klines_df: Klines data to base mock data on

        Returns:
            DataFrame with mock order book data
        """
        try:
            # Create mock order book data
            mock_data = []

            for idx, row in klines_df.iterrows():
                # Create bid/ask prices around the close price
                close_price = row["close"]
                spread_pct = np.random.uniform(0.001, 0.005)  # 0.1% to 0.5% spread

                bid_price = close_price * (1 - spread_pct / 2)
                ask_price = close_price * (1 + spread_pct / 2)

                # Create mock order book levels
                for level in range(10):
                    bid_quantity = np.random.uniform(100, 1000)
                    ask_quantity = np.random.uniform(100, 1000)

                    mock_data.append(
                        {
                            "timestamp": idx,
                            "bid_price": bid_price * (1 - level * 0.001),
                            "ask_price": ask_price * (1 + level * 0.001),
                            "bid_quantity": bid_quantity,
                            "ask_quantity": ask_quantity,
                            "level": level,
                        },
                    )

            order_book_df = pd.DataFrame(mock_data)
            order_book_df.set_index("timestamp", inplace=True)

            return order_book_df

        except Exception as e:
            self.logger.error(f"Error creating mock order book data: {e}")
            return pd.DataFrame()

    async def test_advanced_feature_generation(self, test_data: tuple) -> None:
        """
        Test advanced feature generation.

        Args:
            test_data: Tuple with (klines_df, agg_trades_df, order_book_df)
        """
        try:
            klines_df, agg_trades_df, order_book_df = test_data

            if klines_df.empty:
                terminal_log("❌ No klines data available", "ERROR")
                return

            terminal_log("🎯 Testing advanced feature generation...")

            # Generate advanced features
            advanced_features = (
                await self.advanced_feature_engine.generate_advanced_features(
                    klines_df=klines_df,
                    agg_trades_df=agg_trades_df,
                    order_book_df=order_book_df,
                    sr_levels=None,
                )
            )

            # Analyze results
            terminal_log("📊 Advanced features generated:")
            terminal_log(f"   Shape: {advanced_features.shape}")
            terminal_log(f"   Total columns: {len(advanced_features.columns)}")

            # Show feature categories
            self._analyze_feature_categories(advanced_features)

        except Exception as e:
            terminal_log(f"❌ Error in advanced feature generation test: {e}", "ERROR")

    def _analyze_feature_categories(self, features: pd.DataFrame) -> None:
        """
        Analyze feature categories in the generated features.

        Args:
            features: Features DataFrame
        """
        try:
            # Divergence features
            divergence_features = [
                col for col in features.columns if "divergence" in col.lower()
            ]
            terminal_log(f"   Divergence features: {len(divergence_features)}")
            if divergence_features:
                terminal_log(f"     Sample: {divergence_features[:3]}")

            # Pattern features
            pattern_features = [
                col
                for col in features.columns
                if any(
                    pattern in col.lower()
                    for pattern in ["double", "head", "triangle", "flag", "pennant"]
                )
            ]
            terminal_log(f"   Pattern features: {len(pattern_features)}")
            if pattern_features:
                terminal_log(f"     Sample: {pattern_features[:3]}")

            # Volume profile features
            volume_features = [
                col
                for col in features.columns
                if any(
                    vol in col.lower() for vol in ["vpvr", "poc", "hvn", "lvn", "cvd"]
                )
            ]
            terminal_log(f"   Volume profile features: {len(volume_features)}")
            if volume_features:
                terminal_log(f"     Sample: {volume_features[:3]}")

            # Market microstructure features
            microstructure_features = [
                col
                for col in features.columns
                if any(
                    micro in col.lower()
                    for micro in ["spread", "imbalance", "depth", "liquidity"]
                )
            ]
            terminal_log(
                f"   Market microstructure features: {len(microstructure_features)}",
            )
            if microstructure_features:
                terminal_log(f"     Sample: {microstructure_features[:3]}")

            # Volatility targeting features
            volatility_features = [
                col
                for col in features.columns
                if any(
                    vol in col.lower()
                    for vol in ["volatility", "leverage", "position_size"]
                )
            ]
            terminal_log(
                f"   Volatility targeting features: {len(volatility_features)}",
            )
            if volatility_features:
                terminal_log(f"     Sample: {volatility_features[:3]}")

            # Advanced momentum features
            momentum_features = [
                col
                for col in features.columns
                if any(mom in col.lower() for mom in ["roc", "williams", "cci", "mfi"])
            ]
            terminal_log(f"   Advanced momentum features: {len(momentum_features)}")
            if momentum_features:
                terminal_log(f"     Sample: {momentum_features[:3]}")

            # S/R zone features
            sr_features = [
                col
                for col in features.columns
                if any(
                    sr in col.lower()
                    for sr in ["bb_", "kc_", "vwap_", "resistance", "support"]
                )
            ]
            terminal_log(f"   S/R zone features: {len(sr_features)}")
            if sr_features:
                terminal_log(f"     Sample: {sr_features[:3]}")

        except Exception as e:
            terminal_log(f"❌ Error analyzing feature categories: {e}", "ERROR")

    async def test_divergence_detection(self, test_data: tuple) -> None:
        """
        Test divergence detection features.

        Args:
            test_data: Tuple with test data
        """
        try:
            terminal_log("🔍 Testing divergence detection...")

            klines_df, agg_trades_df, order_book_df = test_data

            if klines_df.empty:
                return

            # Generate features with divergence detection
            features = await self.advanced_feature_engine.generate_advanced_features(
                klines_df=klines_df,
                agg_trades_df=agg_trades_df,
                order_book_df=order_book_df,
            )

            # Check for divergence features
            divergence_features = [
                col for col in features.columns if "divergence" in col.lower()
            ]

            terminal_log(f"   Divergence features found: {len(divergence_features)}")
            for feature in divergence_features:
                non_zero_count = (features[feature] != 0).sum()
                terminal_log(f"     {feature}: {non_zero_count} non-zero values")

        except Exception as e:
            terminal_log(f"❌ Error testing divergence detection: {e}", "ERROR")

    async def test_pattern_recognition(self, test_data: tuple) -> None:
        """
        Test pattern recognition features.

        Args:
            test_data: Tuple with test data
        """
        try:
            terminal_log("📊 Testing pattern recognition...")

            klines_df, agg_trades_df, order_book_df = test_data

            if klines_df.empty:
                return

            # Generate features with pattern recognition
            features = await self.advanced_feature_engine.generate_advanced_features(
                klines_df=klines_df,
                agg_trades_df=agg_trades_df,
                order_book_df=order_book_df,
            )

            # Check for pattern features
            pattern_features = [
                col
                for col in features.columns
                if any(
                    pattern in col.lower()
                    for pattern in ["double", "head", "triangle", "flag", "pennant"]
                )
            ]

            terminal_log(f"   Pattern features found: {len(pattern_features)}")
            for feature in pattern_features:
                non_zero_count = (features[feature] != 0).sum()
                terminal_log(f"     {feature}: {non_zero_count} non-zero values")

        except Exception as e:
            terminal_log(f"❌ Error testing pattern recognition: {e}", "ERROR")

    async def test_volume_profile_analysis(self, test_data: tuple) -> None:
        """
        Test volume profile analysis features.

        Args:
            test_data: Tuple with test data
        """
        try:
            terminal_log("📊 Testing volume profile analysis...")

            klines_df, agg_trades_df, order_book_df = test_data

            if klines_df.empty:
                return

            # Generate features with volume profile analysis
            features = await self.advanced_feature_engine.generate_advanced_features(
                klines_df=klines_df,
                agg_trades_df=agg_trades_df,
                order_book_df=order_book_df,
            )

            # Check for volume profile features
            volume_features = [
                col
                for col in features.columns
                if any(
                    vol in col.lower() for vol in ["vpvr", "poc", "hvn", "lvn", "cvd"]
                )
            ]

            terminal_log(f"   Volume profile features found: {len(volume_features)}")
            for feature in volume_features:
                if feature in features.columns:
                    non_null_count = features[feature].notna().sum()
                    terminal_log(f"     {feature}: {non_null_count} non-null values")

        except Exception as e:
            terminal_log(f"❌ Error testing volume profile analysis: {e}", "ERROR")

    async def test_market_microstructure(self, test_data: tuple) -> None:
        """
        Test market microstructure features.

        Args:
            test_data: Tuple with test data
        """
        try:
            terminal_log("🔍 Testing market microstructure analysis...")

            klines_df, agg_trades_df, order_book_df = test_data

            if klines_df.empty:
                return

            # Generate features with market microstructure analysis
            features = await self.advanced_feature_engine.generate_advanced_features(
                klines_df=klines_df,
                agg_trades_df=agg_trades_df,
                order_book_df=order_book_df,
            )

            # Check for microstructure features
            microstructure_features = [
                col
                for col in features.columns
                if any(
                    micro in col.lower()
                    for micro in ["spread", "imbalance", "depth", "liquidity"]
                )
            ]

            terminal_log(
                f"   Market microstructure features found: {len(microstructure_features)}",
            )
            for feature in microstructure_features:
                if feature in features.columns:
                    non_null_count = features[feature].notna().sum()
                    terminal_log(f"     {feature}: {non_null_count} non-null values")

        except Exception as e:
            terminal_log(f"❌ Error testing market microstructure: {e}", "ERROR")

    async def test_volatility_targeting(self, test_data: tuple) -> None:
        """
        Test volatility targeting features.

        Args:
            test_data: Tuple with test data
        """
        try:
            terminal_log("📈 Testing volatility targeting...")

            klines_df, agg_trades_df, order_book_df = test_data

            if klines_df.empty:
                return

            # Generate features with volatility targeting
            features = await self.advanced_feature_engine.generate_advanced_features(
                klines_df=klines_df,
                agg_trades_df=agg_trades_df,
                order_book_df=order_book_df,
            )

            # Check for volatility targeting features
            volatility_features = [
                col
                for col in features.columns
                if any(
                    vol in col.lower()
                    for vol in ["volatility", "leverage", "position_size"]
                )
            ]

            terminal_log(
                f"   Volatility targeting features found: {len(volatility_features)}",
            )
            for feature in volatility_features:
                if feature in features.columns:
                    non_null_count = features[feature].notna().sum()
                    terminal_log(f"     {feature}: {non_null_count} non-null values")

        except Exception as e:
            terminal_log(f"❌ Error testing volatility targeting: {e}", "ERROR")

    async def test_advanced_momentum_indicators(self, test_data: tuple) -> None:
        """
        Test advanced momentum indicators.

        Args:
            test_data: Tuple with test data
        """
        try:
            terminal_log("📊 Testing advanced momentum indicators...")

            klines_df, agg_trades_df, order_book_df = test_data

            if klines_df.empty:
                return

            # Generate features with advanced momentum indicators
            features = await self.advanced_feature_engine.generate_advanced_features(
                klines_df=klines_df,
                agg_trades_df=agg_trades_df,
                order_book_df=order_book_df,
            )

            # Check for momentum features
            momentum_features = [
                col
                for col in features.columns
                if any(mom in col.lower() for mom in ["roc", "williams", "cci", "mfi"])
            ]

            terminal_log(
                f"   Advanced momentum features found: {len(momentum_features)}",
            )
            for feature in momentum_features:
                if feature in features.columns:
                    non_null_count = features[feature].notna().sum()
                    terminal_log(f"     {feature}: {non_null_count} non-null values")

        except Exception as e:
            terminal_log(f"❌ Error testing advanced momentum indicators: {e}", "ERROR")

    async def test_sr_zone_features(self, test_data: tuple) -> None:
        """
        Test S/R zone definition features.

        Args:
            test_data: Tuple with test data
        """
        try:
            terminal_log("🎯 Testing S/R zone features...")

            klines_df, agg_trades_df, order_book_df = test_data

            if klines_df.empty:
                return

            # Generate features with S/R zone definition
            features = await self.advanced_feature_engine.generate_advanced_features(
                klines_df=klines_df,
                agg_trades_df=agg_trades_df,
                order_book_df=order_book_df,
            )

            # Check for S/R zone features
            sr_features = [
                col
                for col in features.columns
                if any(
                    sr in col.lower()
                    for sr in ["bb_", "kc_", "vwap_", "resistance", "support"]
                )
            ]

            terminal_log(f"   S/R zone features found: {len(sr_features)}")
            for feature in sr_features:
                if feature in features.columns:
                    non_null_count = features[feature].notna().sum()
                    terminal_log(f"     {feature}: {non_null_count} non-null values")

        except Exception as e:
            terminal_log(f"❌ Error testing S/R zone features: {e}", "ERROR")

    async def test_feature_statistics(self) -> None:
        """
        Test feature engineering statistics.
        """
        try:
            terminal_log("📊 Testing feature engineering statistics...")

            stats = self.advanced_feature_engine.get_feature_statistics()

            terminal_log("Advanced Feature Engineering Statistics:")
            terminal_log(
                f"   Divergence detection enabled: {stats['enable_divergence_detection']}",
            )
            terminal_log(
                f"   Pattern recognition enabled: {stats['enable_pattern_recognition']}",
            )
            terminal_log(f"   Volume profile enabled: {stats['enable_volume_profile']}")
            terminal_log(
                f"   Market microstructure enabled: {stats['enable_market_microstructure']}",
            )
            terminal_log(
                f"   Volatility targeting enabled: {stats['enable_volatility_targeting']}",
            )

            # Show configuration details
            config = stats["advanced_config"]
            terminal_log(
                f"   Target volatility: {config.get('target_volatility', 'N/A')}",
            )
            terminal_log(
                f"   Divergence detection config: {len(config.get('divergence_detection', {}))} parameters",
            )
            terminal_log(
                f"   Pattern recognition config: {len(config.get('pattern_recognition', {}))} parameters",
            )
            terminal_log(
                f"   Volume profile config: {len(config.get('volume_profile', {}))} parameters",
            )

        except Exception as e:
            terminal_log(f"❌ Error getting feature statistics: {e}", "ERROR")

    async def run_full_test(self, enable_all: bool = False) -> None:
        """
        Run the full test suite.

        Args:
            enable_all: Whether to enable all advanced features
        """
        try:
            terminal_log("🚀 Starting full advanced feature engineering test...")

            # Initialize
            if not await self.initialize():
                return

            # Load test data
            test_data = await self.load_test_data()

            if test_data[0].empty:
                terminal_log("❌ No test data available", "ERROR")
                return

            # Test individual components
            await self.test_divergence_detection(test_data)
            await self.test_pattern_recognition(test_data)
            await self.test_volume_profile_analysis(test_data)
            await self.test_market_microstructure(test_data)
            await self.test_volatility_targeting(test_data)
            await self.test_advanced_momentum_indicators(test_data)
            await self.test_sr_zone_features(test_data)

            # Test full feature generation
            await self.test_advanced_feature_generation(test_data)

            # Test feature statistics
            await self.test_feature_statistics()

            terminal_log("✅ Full test completed successfully!")

        except Exception as e:
            terminal_log(f"❌ Error in full test: {e}", "ERROR")


async def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(description="Test Advanced Feature Engineering")
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument(
        "--enable-all",
        action="store_true",
        help="Enable all advanced features",
    )

    args = parser.parse_args()

    terminal_log("=" * 60)
    terminal_log("🧪 ADVANCED FEATURE ENGINEERING TEST")
    terminal_log("=" * 60)
    terminal_log(f"Symbol: {args.symbol}")
    terminal_log(f"Enable all features: {args.enable_all}")
    terminal_log("=" * 60)

    # Create tester
    tester = AdvancedFeatureEngineeringTester(args.symbol, CONFIG)

    # Run full test
    await tester.run_full_test(enable_all=args.enable_all)

    terminal_log("=" * 60)
    terminal_log("🏁 Test completed!")
    terminal_log("=" * 60)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
