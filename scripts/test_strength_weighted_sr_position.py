#!/usr/bin/env python3
"""
Test script for strength-weighted SR position calculation.

This script demonstrates a strength-weighted SR position calculation,
showing the difference between simple nearest-level and a strength-weighted approach.
"""


from pathlib import Path
from typing import Any, Dict, List, Tuple
import asyncio
import sys

import numpy as np
import pandas as pd

from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.training.steps.vectorized_advanced_feature_engineering import (
    VectorizedAdvancedFeatureEngineering,
)
from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


class StrengthWeightedSRPositionTester:
    """Test class for strength-weighted SR position calculation."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild("StrengthWeightedSRPositionTester")

        # Basic configuration
        self.config: Dict[str, Any] = {
            "sr_breakout_predictor": {
                "enable_sr_breakout_tactics": True,
                "sr_proximity_threshold": 0.02,
                "breakout_confidence_threshold": 0.7,
                "feature_calculation": {
                    "enable_comprehensive_features": True,
                    "strength_score_weights": {
                        "touch_count": 0.3,
                        "total_volume": 0.2,
                        "level_age": 0.2,
                        "bounce_rate": 0.2,
                        "isolation_score": 0.1,
                    },
                },
            },
            "vectorized_advanced_features": {
                "enable_difference_acceleration_features": True,
            },
        }

        # Initialize components
        self.sr_predictor = SRBreakoutPredictor(self.config)
        self.feature_engine = VectorizedAdvancedFeatureEngineering(self.config)

    @handle_errors(default_return=False, context="tester_initialize")
    def create_sample_price_data(self, periods: int = 1000) -> pd.DataFrame:
        """Create sample OHLCV price data for testing."""
        # Create realistic price data with trends and volatility
        np.random.seed(42)
        trend = np.linspace(100.0, 120.0, periods)
        noise = np.random.normal(0.0, 2.0, periods)
        volatility_cycles = 5.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, periods))
        close_prices = trend + noise + volatility_cycles

        data: List[Dict[str, float]] = []
        for close in close_prices:
            high = close + abs(float(np.random.normal(0.0, 1.0)))
            low = close - abs(float(np.random.normal(0.0, 1.0)))
            open_price = close + float(np.random.normal(0.0, 0.5))
            volume = float(np.random.uniform(1000, 10000))
            data.append({
                "open": open_price,
                "high": high,
                "low": low,
                "close": float(close),
                "volume": volume,
            })

        df = pd.DataFrame(data)
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="1H")
        self.logger.info(f"Created sample price data with {len(df)} periods")
        return df

    def create_sample_sr_levels(self, price_data: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
        """Create sample SR levels with different strengths for testing."""
        close = price_data["close"]
        current_price = float(close.iloc[-1])

        # Support and resistance levels with varying strengths
        support_levels: List[Dict[str, float]] = [
            {"price": current_price * 0.95, "strength": 0.9, "touches": 5, "volume": 50000, "age": 30},
            {"price": current_price * 0.97, "strength": 0.6, "touches": 3, "volume": 25000, "age": 15},
            {"price": current_price * 0.90, "strength": 0.3, "touches": 1, "volume": 5000,  "age": 5},
        ]
        resistance_levels: List[Dict[str, float]] = [
            {"price": current_price * 1.05, "strength": 0.8, "touches": 4, "volume": 40000, "age": 25},
            {"price": current_price * 1.02, "strength": 0.5, "touches": 2, "volume": 15000, "age": 10},
            {"price": current_price * 1.10, "strength": 0.2, "touches": 1, "volume": 3000,  "age": 3},
        ]

        self.logger.info(
            f"Created sample SR levels: {len(support_levels)} support, {len(resistance_levels)} resistance"
        )
        return {"support_levels": support_levels, "resistance_levels": resistance_levels}

    def calculate_simple_position(self, price: float, sr_levels: Dict[str, List[Dict[str, float]]]) -> float:
        """Calculate simple position using nearest support and resistance."""
        support_levels = sr_levels.get("support_levels", [])
        resistance_levels = sr_levels.get("resistance_levels", [])

        nearest_support: float | None = None
        nearest_resistance: float | None = None

        for level in support_levels:
            level_price = float(level.get("price", 0.0))
            if level_price > 0.0 and level_price < price:
                if nearest_support is None or abs(price - level_price) < abs(price - nearest_support):
                    nearest_support = level_price

        for level in resistance_levels:
            level_price = float(level.get("price", 0.0))
            if level_price > 0.0 and level_price > price:
                if nearest_resistance is None or abs(price - level_price) < abs(price - nearest_resistance):
                    nearest_resistance = level_price

        if nearest_support is not None and nearest_resistance is not None:
            if nearest_resistance == nearest_support:
                return 0.5
            position = (price - nearest_support) / (nearest_resistance - nearest_support)
            return float(max(0.0, min(1.0, position)))

        return 0.5

    def calculate_strength_weighted_position(self, price: float, sr_levels: Dict[str, List[Dict[str, float]]]) -> Tuple[float, Dict[str, float]]:
        """Calculate a strength-weighted SR position and return position plus details."""
        support_scores: List[Tuple[float, float]] = []
        resistance_scores: List[Tuple[float, float]] = []

        # Support scores
        for level in sr_levels.get("support_levels", []):
            level_price = float(level["price"])
            if level_price < price:
                distance = (price - level_price) / price
                proximity_factor = float(np.exp(-50.0 * distance))
                final_score = float(level.get("strength", 0.5)) * proximity_factor
                support_scores.append((level_price, final_score))

        # Resistance scores
        for level in sr_levels.get("resistance_levels", []):
            level_price = float(level["price"])
            if level_price > price:
                distance = (level_price - price) / price
                proximity_factor = float(np.exp(-50.0 * distance))
                final_score = float(level.get("strength", 0.5)) * proximity_factor
                resistance_scores.append((level_price, final_score))

        effective_support = max(support_scores, key=lambda x: x[1]) if support_scores else None
        effective_resistance = max(resistance_scores, key=lambda x: x[1]) if resistance_scores else None

        details: Dict[str, float] = {}
        if effective_support and effective_resistance:
            support_price, support_score = effective_support
            resistance_price, resistance_score = effective_resistance
            if resistance_price == support_price:
                return 0.5, {"support_price": support_price, "support_score": support_score, "resistance_price": resistance_price, "resistance_score": resistance_score}
            strength_position = (price - support_price) / (resistance_price - support_price)
            strength_position = float(max(0.0, min(1.0, strength_position)))
            details = {
                "support_price": float(support_price),
                "support_score": float(support_score),
                "resistance_price": float(resistance_price),
                "resistance_score": float(resistance_score),
            }
            return strength_position, details

        return 0.5, {}

    @handle_errors(default_return=False, context="tester_run")
    async def test_strength_weighted_position(self) -> bool:
        self.logger.info("Starting strength-weighted SR position test...")

        # Create sample data
        price_data = self.create_sample_price_data(100)
        if price_data.empty:
            self.logger.warning("Empty price data")
            return False

        # Create sample SR levels
        sr_levels = self.create_sample_sr_levels(price_data)
        if not sr_levels:
            self.logger.warning("No SR levels available")
            return False

        # Test with different price points
        test_prices = [
            float(price_data["close"].iloc[-1]),
            float(price_data["close"].iloc[-1] * 0.96),
            float(price_data["close"].iloc[-1] * 1.03),
            float(price_data["close"].iloc[-1] * 0.92),
            float(price_data["close"].iloc[-1] * 1.08),
        ]

        self.logger.info("=" * 80)
        self.logger.info("STRENGTH-WEIGHTED SR POSITION ANALYSIS")
        self.logger.info("=" * 80)

        for i, test_price in enumerate(test_prices, start=1):
            self.logger.info(f"Test Price {i}: ${test_price:.2f}")

            # Calculate simple position
            simple_position = self.calculate_simple_position(test_price, sr_levels)

            # Calculate strength-weighted position
            strength_position, details = self.calculate_strength_weighted_position(test_price, sr_levels)

            self.logger.info(f"   Simple Position: {simple_position:.3f}")
            self.logger.info(f"   Strength Position: {strength_position:.3f}")
            if details:
                self.logger.info(
                    f"   Effective Support: ${details['support_price']:.2f} (score: {details['support_score']:.3f})"
                )
                self.logger.info(
                    f"   Effective Resistance: ${details['resistance_price']:.2f} (score: {details['resistance_score']:.3f})"
                )

            # Show the difference
            position_diff = abs(strength_position - simple_position)
            self.logger.info(f"   Position Difference: {position_diff:.3f}")
            if position_diff > 0.1:
                self.logger.info("   Significant difference detected!")
            else:
                self.logger.info("   Positions are similar")

        # Show feature summary from predictor (optional)
        try:
            features = await self.sr_predictor.calculate_sr_features(price_data)
            if isinstance(features, dict):
                self.logger.info(f"Generated Features: {len(features)}")
                self.logger.info("Feature names:")
                for feature_name in sorted(features.keys()):
                    self.logger.info(f"   - {feature_name}")
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Feature generation skipped due to error: {e}")

        self.logger.info("Strength-weighted SR position test completed successfully")
        return True


@handle_errors(default_return=None, context="tester_main")
async def main() -> None:
    tester = StrengthWeightedSRPositionTester()
    if await tester.initialize():
        await tester.test_strength_weighted_position()
    else:
        print("Failed to initialize tester")


if __name__ == "__main__":
    asyncio.run(main())
