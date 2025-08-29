#!/usr/bin/env python3
"""
Enhanced SR Position Analyzer

This script analyzes the position of current price between the closest support and resistance levels,
using the centralized SRBreakoutPredictor for professional-grade S/R analysis.

Features:
- Uses centralized SRBreakoutPredictor for S/R level detection
- Enhanced strength calculation with DBSCAN clustering
- Advanced S/R methods (Fibonacci, Elliott Wave, Order Flow)
- Multi-factor strength scoring
- Professional noise filtering

Usage:
    python scripts/analyze_sr_position_enhanced.py --symbol ETHUSDT --exchange BINANCE --timeframe 15m
"""

from pathlib import Path
import argparse
import asyncio
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.logger import system_logger


class EnhancedSRPositionAnalyzer:
    """
    Enhanced analyzer for calculating position between support and resistance levels
    using the centralized SRBreakoutPredictor.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EnhancedSRPositionAnalyzer")
        self.sr_predictor = None

    async def initialize(self) -> bool:
        """Initialize the analyzer with SRBreakoutPredictor."""
        self.logger.info("🚀 Initializing Enhanced SR Position Analyzer...")

        # Initialize SRBreakoutPredictor with enhanced configuration
        sr_config = {
            "sr_breakout_predictor": {
                "enable_sr_breakout_tactics": True,
                "sr_proximity_threshold": 0.02,
                "breakout_confidence_threshold": 0.6,
                "sr_detection_method": "fractal",
                "min_sr_strength": 0.3,
                "max_sr_levels": 10,
                "sr_lookback_periods": 100,
                "volume_weight": 0.7,
                "price_weight": 0.3,
                "atr_multiplier": 1.5,
                "breakout_confirmation_periods": 3,
                "false_breakout_filter": True,
                
                # Enhanced strength calculation configuration
                "strength_calculation": {
                    "enable_enhanced_strength": True,
                    "touch_count_lookback": 50,
                    "bounce_rate_threshold": 0.02,
                    "isolation_distance_threshold": 0.05,
                    "age_decay_factor": 0.95
                },
                
                # DBSCAN clustering configuration
                "dbscan_clustering": {
                    "enable_dbscan_clustering": True,
                    "eps": 0.01,
                    "min_samples": 2,
                    "enable_noise_filtering": True
                },
                
                # Feature calculation configuration
                "feature_calculation": {
                    "enable_comprehensive_features": True,
                    "strength_score_weights": {
                        "touch_count": 0.3,
                        "total_volume": 0.2,
                        "level_age": 0.2,
                        "bounce_rate": 0.2,
                        "isolation_score": 0.1
                    }
                }
            }
        }

        self.sr_predictor = SRBreakoutPredictor(sr_config)
        init_success = await self.sr_predictor.initialize()
        
        if not init_success:
            self.logger.error("❌ Failed to initialize SRBreakoutPredictor")
            return False

        self.logger.info("✅ Enhanced SR Position Analyzer initialized successfully")
        return True

    def calculate_sr_position(self, price_data: pd.DataFrame, sr_context: Dict[str, Any]) -> pd.Series:
        """
        Calculate position between closest support and resistance levels using enhanced S/R analysis.

        Args:
            price_data: OHLCV price data
            sr_context: S/R context from SRBreakoutPredictor

        Returns:
            Series with position values from 0 (at support) to 1 (at resistance)
        """
        if price_data.empty or "close" not in price_data.columns:
            self.logger.warning("⚠️ Invalid price data for SR position calculation")
            return pd.Series(dtype=float)

        close = price_data["close"].astype(float)
        
        # Get S/R levels from context
        support_levels = sr_context.get("support_levels", [])
        resistance_levels = sr_context.get("resistance_levels", [])
        
        # Get nearest levels
        nearest_support = sr_context.get("nearest_support", close.iloc[-1])
        nearest_resistance = sr_context.get("nearest_resistance", close.iloc[-1])

        if not support_levels or not resistance_levels:
            self.logger.warning("⚠️ No SR levels available for position calculation")
            return pd.Series(dtype=float)

        positions: list[float] = []

        for price in close:
            if pd.isna(price):
                positions.append(np.nan)
                continue

            # Find closest support and resistance levels
            support_distances = [abs(price - level.get('price', level)) for level in support_levels if level.get('price', level) > 0]
            resistance_distances = [abs(price - level.get('price', level)) for level in resistance_levels if level.get('price', level) > 0]

            if not support_distances or not resistance_distances:
                positions.append(0.5)  # Default to middle if no levels found
                continue

            # Find the actual closest support and resistance levels
            closest_support = min(support_levels, key=lambda x: abs(price - x.get('price', x)))
            closest_resistance = min(resistance_levels, key=lambda x: abs(price - x.get('price', x)))
            
            support_price = closest_support.get('price', closest_support)
            resistance_price = closest_resistance.get('price', closest_resistance)

            # Ensure support is below and resistance is above current price
            if support_price > price:
                # If closest support is above price, find the next support below
                supports_below = [s for s in support_levels if s.get('price', s) < price]
                if supports_below:
                    support_price = max(supports_below, key=lambda x: x.get('price', x)).get('price', max(supports_below, key=lambda x: x.get('price', x)))
                else:
                    support_price = price * 0.95  # Default 5% below

            if resistance_price < price:
                # If closest resistance is below price, find the next resistance above
                resistances_above = [r for r in resistance_levels if r.get('price', r) > price]
                if resistances_above:
                    resistance_price = min(resistances_above, key=lambda x: x.get('price', x)).get('price', min(resistances_above, key=lambda x: x.get('price', x)))
                else:
                    resistance_price = price * 1.05  # Default 5% above

            # Calculate position (0 = at support, 1 = at resistance)
            if resistance_price == support_price:
                position = 0.5  # At the same level
            else:
                position = (price - support_price) / (resistance_price - support_price)
                position = max(0.0, min(1.0, position))  # Clamp to [0, 1]

            positions.append(position)

        return pd.Series(positions, index=price_data.index)

    def analyze_sr_position(self, price_data: pd.DataFrame, sr_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze SR position with enhanced metrics.

        Args:
            price_data: OHLCV price data
            sr_context: S/R context from SRBreakoutPredictor

        Returns:
            Dictionary with comprehensive analysis results
        """
        self.logger.info("🔍 Analyzing SR position with enhanced metrics...")

        # Calculate position series
        position_series = self.calculate_sr_position(price_data, sr_context)
        
        if position_series.empty:
            return {}

        # Basic statistics
        current_position = position_series.iloc[-1]
        mean_position = position_series.mean()
        std_position = position_series.std()

        # Zone analysis
        near_support = (position_series <= 0.2).sum()
        near_resistance = (position_series >= 0.8).sum()
        middle_zone = ((position_series > 0.2) & (position_series < 0.8)).sum()

        # Trend analysis
        position_trend = position_series.diff().fillna(0)
        trend_direction = "upward" if position_trend.iloc[-10:].mean() > 0 else "downward"
        
        # Volatility analysis
        position_volatility = position_series.rolling(20).std().fillna(0)
        current_volatility = position_volatility.iloc[-1]

        # Enhanced S/R metrics
        enhanced_support = sr_context.get("enhanced_strength_support", {})
        enhanced_resistance = sr_context.get("enhanced_strength_resistance", {})
        clustering_result = sr_context.get("clustering_result", {})
        
        # Count significant levels
        significant_support = len([level for level in sr_context.get("support_levels", []) 
                                 if level.get("enhanced_strength", 0.5) > 0.6])
        significant_resistance = len([level for level in sr_context.get("resistance_levels", []) 
                                    if level.get("enhanced_strength", 0.5) > 0.6])

        analysis = {
            "position_metrics": {
                "current_position": current_position,
                "mean_position": mean_position,
                "std_position": std_position,
                "position_range": (position_series.min(), position_series.max()),
            },
            "zone_analysis": {
                "near_support_count": near_support,
                "near_resistance_count": near_resistance,
                "middle_zone_count": middle_zone,
                "total_periods": len(position_series),
                "support_zone_percentage": (near_support / len(position_series)) * 100,
                "resistance_zone_percentage": (near_resistance / len(position_series)) * 100,
                "middle_zone_percentage": (middle_zone / len(position_series)) * 100,
            },
            "trend_analysis": {
                "trend_direction": trend_direction,
                "trend_strength": abs(position_trend.iloc[-10:].mean()),
                "position_volatility": current_volatility,
                "volatility_trend": position_volatility.iloc[-10:].mean(),
            },
            "enhanced_sr_metrics": {
                "significant_support_levels": significant_support,
                "significant_resistance_levels": significant_resistance,
                "total_clusters": clustering_result.get("n_clusters", 0),
                "noise_filtered": clustering_result.get("noise_points", 0),
                "support_strength": sr_context.get("support_strength", 0.5),
                "resistance_strength": sr_context.get("resistance_strength", 0.5),
                "sr_zone_width": sr_context.get("sr_zone_width", 0.0),
            },
            "sr_levels": {
                "support_levels": sr_context.get("support_levels", []),
                "resistance_levels": sr_context.get("resistance_levels", []),
                "position_series": position_series,
                "enhanced_strength_support": enhanced_support,
                "enhanced_strength_resistance": enhanced_resistance,
                "clustering_result": clustering_result,
            }
        }

        self.logger.info("✅ SR position analysis completed")
        return analysis

    def print_analysis_report(self, analysis: Dict[str, Any]) -> None:
        """Print comprehensive analysis report."""
        if not analysis:
            print("❌ No analysis results to display")
            return

        print("\n" + "=" * 80)
        print("📊 ENHANCED SR POSITION ANALYSIS REPORT")
        print("=" * 80)

        # Position metrics
        pos_metrics = analysis["position_metrics"]
        print(f"\n📍 POSITION METRICS:")
        print(f"   Current Position: {pos_metrics['current_position']:.3f} (0=Support, 1=Resistance)")
        print(f"   Mean Position: {pos_metrics['mean_position']:.3f}")
        print(f"   Position Std Dev: {pos_metrics['std_position']:.3f}")
        print(f"   Position Range: {pos_metrics['position_range'][0]:.3f} - {pos_metrics['position_range'][1]:.3f}")

        # Zone analysis
        zone_analysis = analysis["zone_analysis"]
        print(f"\n🎯 ZONE ANALYSIS:")
        print(f"   Near Support: {zone_analysis['near_support_count']} periods ({zone_analysis['support_zone_percentage']:.1f}%)")
        print(f"   Near Resistance: {zone_analysis['near_resistance_count']} periods ({zone_analysis['resistance_zone_percentage']:.1f}%)")
        print(f"   Middle Zone: {zone_analysis['middle_zone_count']} periods ({zone_analysis['middle_zone_percentage']:.1f}%)")

        # Trend analysis
        trend_analysis = analysis["trend_analysis"]
        print(f"\n📈 TREND ANALYSIS:")
        print(f"   Trend Direction: {trend_analysis['trend_direction']}")
        print(f"   Trend Strength: {trend_analysis['trend_strength']:.3f}")
        print(f"   Position Volatility: {trend_analysis['position_volatility']:.3f}")

        # Enhanced S/R metrics
        enhanced_metrics = analysis["enhanced_sr_metrics"]
        print(f"\n💪 ENHANCED S/R METRICS:")
        print(f"   Significant Support Levels: {enhanced_metrics['significant_support_levels']}")
        print(f"   Significant Resistance Levels: {enhanced_metrics['significant_resistance_levels']}")
        print(f"   Total Clusters: {enhanced_metrics['total_clusters']}")
        print(f"   Noise Filtered: {enhanced_metrics['noise_filtered']}")
        print(f"   Support Strength: {enhanced_metrics['support_strength']:.3f}")
        print(f"   Resistance Strength: {enhanced_metrics['resistance_strength']:.3f}")
        print(f"   S/R Zone Width: {enhanced_metrics['sr_zone_width']:.3f}")

        # S/R Levels summary
        sr_levels = analysis["sr_levels"]
        print(f"\n🎯 S/R LEVELS SUMMARY:")
        print(f"   Support Levels: {len(sr_levels['support_levels'])} levels")
        if sr_levels["support_levels"]:
            support_prices = [level.get('price', level) for level in sr_levels['support_levels']]
            print(f"   Support Range: {min(support_prices):.2f} - {max(support_prices):.2f}")
        
        print(f"   Resistance Levels: {len(sr_levels['resistance_levels'])} levels")
        if sr_levels["resistance_levels"]:
            resistance_prices = [level.get('price', level) for level in sr_levels['resistance_levels']]
            print(f"   Resistance Range: {min(resistance_prices):.2f} - {max(resistance_prices):.2f}")

        print("=" * 80)


async def load_price_data(symbol: str, exchange: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load price data for analysis."""
    # Try multiple data formats and locations
    possible_paths = [
        Path(f"data/{exchange.lower()}/{symbol.lower()}_{timeframe}.parquet"),
        Path(f"data/{symbol}_{timeframe}.csv"),
        Path(f"data/{symbol}_{timeframe}.parquet"),
        Path(f"data/{exchange}_{symbol}_labeled_regimes.csv"),
    ]

    for path in possible_paths:
        if path.exists():
            if path.suffix == ".csv":
                return pd.read_csv(path)
            if path.suffix == ".parquet":
                return pd.read_parquet(path)

    return None


async def main():
    """Main function to run enhanced SR position analysis."""
    parser = argparse.ArgumentParser(
        description="Enhanced SR position analysis using centralized SRBreakoutPredictor"
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

    system_logger.info(
        f"🚀 Starting Enhanced SR Position Analysis for {args.symbol} on {args.exchange}"
    )

    # Load price data
    price_data = await load_price_data(args.symbol, args.exchange, args.timeframe)
    if price_data is None:
        system_logger.error("❌ Failed to load price data")
        return

    # Initialize analyzer
    config = {
        "symbol": args.symbol, 
        "exchange": args.exchange,
        "timeframe": args.timeframe
    }

    analyzer = EnhancedSRPositionAnalyzer(config)
    if not await analyzer.initialize():
        system_logger.error("❌ Failed to initialize analyzer")
        return

    # Get current price
    current_price = price_data["close"].iloc[-1]
    
    # Get enhanced S/R context using SRBreakoutPredictor
    sr_context = await analyzer.sr_predictor.get_sr_context(price_data, current_price)
    if not sr_context:
        system_logger.error("❌ Failed to generate S/R context")
        return

    # Perform analysis
    analysis = analyzer.analyze_sr_position(price_data, sr_context)

    # Print report
    analyzer.print_analysis_report(analysis)

    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save position series to CSV
        if "position_series" in analysis["sr_levels"]:
            position_df = pd.DataFrame(
                {
                    "timestamp": analysis["sr_levels"]["position_series"].index,
                    "sr_position": analysis["sr_levels"]["position_series"].values,
                }
            )
            position_df.to_csv(output_path, index=False)
            system_logger.info(f"✅ Saved detailed results to {output_path}")

    system_logger.info("✅ Enhanced SR Position Analysis completed successfully")


if __name__ == "__main__":
    asyncio.run(main())