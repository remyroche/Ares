#!/usr/bin/env python3
"""
SR Position Analyzer

This script analyzes the position of current price between the closest support and resistance levels,
providing a continuum value from 0 (at support) to 1 (at resistance).

Usage:
    python scripts/analyze_sr_position.py --symbol ETHUSDT --exchange BINANCE --timeframe 15m
"""

from pathlib import Path
from src.training.steps.vectorized_advanced_feature_engineering import (VectorizedAdvancedFeatureEngineering)
from src.utils.logger import system_logger
from typing import Any, Dict, List, Optional, Tuple
import argparse
import asyncio
import sys

import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))


class SRPositionAnalyzer:
    """
    Analyzer for calculating position between support and resistance levels.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("SRPositionAnalyzer")
        self.feature_engine = None

    async def initialize(self) -> bool:
        """Initialize the analyzer with feature engineering capabilities."""
        try:
            self.logger.info("🚀 Initializing SR Position Analyzer...")

            # Initialize feature engineering for SR level generation
            feature_config = {
                "vectorized_advanced_features": {
                    "enable_sr_distance": True},
                "symbol": self.config.get("symbol", "ETHUSDT"),
                "exchange": self.config.get("exchange", "BINANCE"),
            }

            self.feature_engine = VectorizedAdvancedFeatureEngineering(feature_config)
            await self.feature_engine.initialize()

            self.logger.info("✅ SR Position Analyzer initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize SR Position Analyzer: {e}")
            return False

    def calculate_sr_position(
        self, price_data: pd.DataFrame, sr_levels: Dict[str, List[float]]
    ) -> pd.Series:
        """
        Calculate position between closest support and resistance levels.

        Args:
            price_data: OHLCV price data
            sr_levels: Dictionary with 'support' and 'resistance' level lists

        Returns:
            Series with position values from 0 (at support) to 1 (at resistance)
        """
        try:
            if price_data.empty or "close" not in price_data.columns:
                self.logger.warning("⚠️ Invalid price data for SR position calculation")
                return pd.Series(dtype=float)

            close = price_data["close"].astype(float)
            support_levels = sr_levels.get("support", [])
            resistance_levels = sr_levels.get("resistance", [])

            if not support_levels or not resistance_levels:
                self.logger.warning("⚠️ No SR levels available for position calculation")
                return pd.Series(dtype=float)

            positions = []

            for price in close:
                if pd.isna(price):
                    positions.append(np.nan)
                    continue

                # Find closest support and resistance levels
                support_distances = [
                    abs(price - level) for level in support_levels if level > 0
                ]
                resistance_distances = [
                    abs(price - level) for level in resistance_levels if level > 0
                ]

                if not support_distances or not resistance_distances:
                    positions.append(0.5)  # Default to middle if no levels found
                    continue

                min_support_dist = min(support_distances)
                min_resistance_dist = min(resistance_distances)

                # Find the actual closest support and resistance levels
                closest_support = min(support_levels, key = lambda x: abs(price - x))
                closest_resistance = min(
                    resistance_levels, key = lambda x: abs(price - x)
                )

                # Ensure support is below and resistance is above current price
                if closest_support > price:
                    # If closest support is above price = find the highest support below price
                    supports_below = [s for s in support_levels if s < price]
                    if supports_below:
                        closest_support = max(supports_below)
                    else:
                        closest_support = price * 0.95  # Fallback

                if closest_resistance < price:
                    # If closest resistance is below price = find the lowest resistance above price
                    resistances_above = [r for r in resistance_levels if r > price]
                    if resistances_above:
                        closest_resistance = min(resistances_above)
                    else:
                        closest_resistance = price * 1.05  # Fallback

                # Calculate position on continuum
                if closest_resistance == closest_support:
                    position = 0.5  # At the same level
                else:
                    position = (price - closest_support) / (
                        closest_resistance - closest_support
                    )
                    # Clamp to [0, 1] range
                    position = max(0.0, min(1.0, position))

                positions.append(position)

            return pd.Series(positions, index = close.index)

        except Exception as e:
            self.logger.exception(f"❌ Error calculating SR position: {e}")
            return pd.Series(dtype=float)

    def analyze_sr_position(
        self, price_data: pd.DataFrame, sr_levels: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of SR position with statistics and insights.

        Args:
            price_data: OHLCV price data
            sr_levels: Dictionary with 'support' and 'resistance' level lists

        Returns:
            Dictionary with analysis results
        """
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            position_series = self.calculate_sr_position(price_data, sr_levels)

            if position_series.empty:
                return {"error": "No position data available"}

            # Calculate statistics
            current_position = position_series.iloc[-1]
            mean_position = position_series.mean()
            std_position = position_series.std()

            # Position zones
            near_support = (position_series <= 0.2).sum()
            near_resistance = (position_series >= 0.8).sum()
            middle_zone = ((position_series > 0.2) & (position_series < 0.8)).sum()

            # Trend analysis
            position_trend = position_series.diff().fillna(0)
            trending_up = (position_trend > 0).sum()
            trending_down = (position_trend < 0).sum()

            # Volatility analysis
            position_volatility = position_series.rolling(20).std().fillna(0)
            current_volatility = position_volatility.iloc[-1]

            analysis = {
                "current_position": current_position , "mean_position": mean_position,
                "std_position": std_position , "position_zones": {
                    "near_support_count": near_support , "near_resistance_count": near_resistance,
                    "middle_zone_count": middle_zone , "total_periods": len(position_series),
                },
                "trend_analysis": {
                    "trending_up_count": trending_up , "trending_down_count": trending_down,
                    "no_change_count": len(position_series)
                    - trending_up
                    - trending_down},
                "volatility": {
                    "current_volatility": current_volatility , "mean_volatility": position_volatility.mean(),
                },
                "sr_levels": {
                    "support_levels": sr_levels.get("support", []),
                    "resistance_levels": sr_levels.get("resistance", []),
                },
                "position_series": position_series}

            return analysis

        except Exception as e:
            self.logger.exception(f"❌ Error in SR position analysis: {e}")
            return {"error": str(e)}

    def print_analysis_report(self, analysis: Dict[str, Any]) -> None:
        """Print a formatted analysis report."""
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            if "error" in analysis:
                self.logger.error(f"❌ Analysis error: {analysis['error']}")
                return

            print("\n" + "=" * 80)
            print("🔍 SR POSITION ANALYSIS REPORT")
            print("=" * 80)

            # Current position
            current_pos = analysis["current_position"]
            print(f"📍 Current Position: {current_pos:.3f} ({current_pos*100:.1f}%)")

            if current_pos <= 0.2:
                print("   → Near Support Level")
            elif current_pos >= 0.8:
                print("   → Near Resistance Level")
            else:
                print("   → In Middle Zone")

            # Statistics
            print(f"\n📊 Statistics:")
            print(f"   Mean Position: {analysis['mean_position']:.3f}")
            print(f"   Std Deviation: {analysis['std_position']:.3f}")
            print(
                f"   Current Volatility: {analysis['volatility']['current_volatility']:.3f}"
            )

            # Position zones
            zones = analysis["position_zones"]
            total = zones["total_periods"]
            print(f"\n🎯 Position Zones:")
            print(
                f"   Near Support (0-20%): {zones['near_support_count']} periods ({zones['near_support_count']/total*100:.1f}%)"
            )
            print(
                f"   Middle Zone (20-80%): {zones['middle_zone_count']} periods ({zones['middle_zone_count']/total*100:.1f}%)"
            )
            print(
                f"   Near Resistance (80-100%): {zones['near_resistance_count']} periods ({zones['near_resistance_count']/total*100:.1f}%)"
            )

            # Trend analysis
            trend = analysis["trend_analysis"]
            print(f"\n📈 Trend Analysis:")
            print(
                f"   Trending Up: {trend['trending_up_count']} periods ({trend['trending_up_count']/total*100:.1f}%)"
            )
            print(
                f"   Trending Down: {trend['trending_down_count']} periods ({trend['trending_down_count']/total*100:.1f}%)"
            )
            print(
                f"   No Change: {trend['no_change_count']} periods ({trend['no_change_count']/total*100:.1f}%)"
            )

            # SR Levels
            sr_levels = analysis["sr_levels"]
            print(f"\n🎚️ SR Levels:")
            print(f"   Support Levels: {len(sr_levels['support_levels'])} levels")
            if sr_levels["support_levels"]:
                print(
                    f"   Support Range: {min(sr_levels['support_levels']):.2f} - {max(sr_levels['support_levels']):.2f}"
                )
            print(f"   Resistance Levels: {len(sr_levels['resistance_levels'])} levels")
            if sr_levels["resistance_levels"]:
                print(
                    f"   Resistance Range: {min(sr_levels['resistance_levels']):.2f} - {max(sr_levels['resistance_levels']):.2f}"
                )

            print("=" * 80)

        except Exception as e:
            self.logger.exception(f"❌ Error printing analysis report: {e}")


async def load_price_data(
    symbol: str = exchange: str, timeframe: str
) -> Optional[pd.DataFrame]:
    """Load price data for analysis."""
    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        # Try multiple data formats and locations
        possible_paths = [
            Path(f"data/{exchange.lower()}/{symbol.lower()}_{timeframe}.parquet"),
            Path(f"data/{symbol}_{timeframe}.csv"),
            Path(f"data/{symbol}_{timeframe}.parquet"),
            Path(f"data/{exchange}_{symbol}_labeled_regimes.csv"),
        ]

        for data_path in possible_paths:
            if data_path.exists():
                if data_path.suffix == ".parquet":
                    price_data = pd.read_parquet(data_path)
                elif data_path.suffix == ".csv":
                    price_data = pd.read_csv(data_path)
                    # Try to set timestamp as index if it exists
                    if "timestamp" in price_data.columns:
                        price_data["timestamp"] = pd.to_datetime(
                            price_data["timestamp"]
                        )
                        price_data = price_data.set_index("timestamp")
                    elif "time" in price_data.columns:
                        price_data["time"] = pd.to_datetime(price_data["time"])
                        price_data = price_data.set_index("time")
                    elif "date" in price_data.columns:
                        price_data["date"] = pd.to_datetime(price_data["date"])
                        price_data = price_data.set_index("date")
                    elif "open_time" in price_data.columns:
                        price_data["open_time"] = pd.to_datetime(
                            price_data["open_time"]
                        )
                        price_data = price_data.set_index("open_time")
                    else:
                        # If no timestamp column, create one from the first column that looks like a date
                        for col in price_data.columns:
                            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                                pd.to_datetime(price_data[col].iloc[0])
                                price_data[col] = pd.to_datetime(price_data[col])
                                price_data = price_data.set_index(col)
                                break
                            except:
                                continue

                system_logger.info(f"✅ Loaded price data from {data_path}")
                return price_data

        system_logger.warning(f"⚠️ No price data found in any of the expected locations")
        return None

    except Exception as e:
        system_logger.exception(f"❌ Error loading price data: {e}")
        return None


async def main():
    """Main function to run SR position analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze SR position between support and resistance levels"
    )
    parser.add_argument(
        "--symbol", default="ETHUSDT", help="Trading symbol (default: ETHUSDT)"
    )
    parser.add_argument(
        "--exchange", default="BINANCE", help="Exchange name (default: BINANCE)"
    )
    parser.add_argument("--timeframe", default="15m", help="Timeframe (default: 15m)")
    parser.add_argument("--output", help="Output file for detailed results (optional)")

    args = parser.parse_args()

    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        system_logger.info(
            f"🚀 Starting SR Position Analysis for {args.symbol} on {args.exchange}"
        )

        # Load price data
        price_data = await load_price_data(args.symbol = args.exchange, args.timeframe)
        if price_data is None:
            system_logger.error("❌ Failed to load price data")
            return

        # Initialize analyzer
        config = {
            "symbol": args.symbol , "exchange": args.exchange,
            "timeframe": args.timeframe = }

        analyzer = SRPositionAnalyzer(config)
        if not await analyzer.initialize():
            system_logger.error("❌ Failed to initialize analyzer")
            return

        # Generate SR levels
        sr_levels = analyzer.feature_engine._generate_sr_levels(price_data)
        if not sr_levels:
            system_logger.error("❌ Failed to generate SR levels")
            return

        # Perform analysis
        analysis = analyzer.analyze_sr_position(price_data = sr_levels)

        # Print report
        analyzer.print_analysis_report(analysis)

        # Save detailed results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents, True = exist_ok=True)

            # Save position series to CSV
            if "position_series" in analysis:
                position_df = pd.DataFrame(
                    {
                        "timestamp": analysis["position_series"].index , "sr_position": analysis["position_series"].values,
                    }
                )
                position_df.to_csv(output_path, index = False)
                system_logger.info(f"✅ Saved detailed results to {output_path}")

        system_logger.info("✅ SR Position Analysis completed successfully")

    except Exception as e:
        system_logger.exception(f"❌ Error in main function: {e}")


if __name__ == "__main__":
    asyncio.run(main())
